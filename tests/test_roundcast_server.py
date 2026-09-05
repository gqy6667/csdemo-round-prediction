import http.client
import json
import socket
import threading
import unittest
from unittest.mock import patch
from uuid import UUID

from src.csdemo.roundcast_service import MODEL_FILES, RoundcastService
from src.csdemo.roundcast_server import create_server


class RoundcastHTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = RoundcastService()
        cls.server = create_server(port=0, service=cls.service)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(5)

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        connection.request(method, path, body, headers or {})
        response = connection.getresponse()
        status, data, response_headers = response.status, response.read(), dict(response.getheaders())
        connection.close()
        return status, data, response_headers

    def test_neutral_details_and_explicit_outcome(self):
        status, data, _ = self.request("GET", "/api/examples/A")
        self.assertEqual(status, 200)
        detail = json.loads(data)
        self.assertEqual(detail["features"]["round_num"], 4)
        for forbidden in (b"winning_side", b"ct_win", b"reference", b"first_kill"):
            self.assertNotIn(forbidden, data)
        self.assertEqual(json.loads(self.request("GET", "/api/examples/A/outcome")[1])["winning_side"], "CT")

    def test_catalogs_expose_four_ready_models_and_three_ready_examples(self):
        models = json.loads(self.request("GET", "/api/models")[1])["models"]
        self.assertEqual(len(models), 4)
        self.assertEqual(
            {(m["stage"], m["algorithm"], m["model_id"]) for m in models},
            {(stage, algorithm, route[0]) for (stage, algorithm), route in MODEL_FILES.items()},
        )
        self.assertTrue(all(m["inference_ready"] for m in models))
        self.assertTrue(all(m["available_examples"] == ["A", "B", "C"] for m in models))
        examples = json.loads(self.request("GET", "/api/examples")[1])["examples"]
        self.assertEqual([e["example_id"] for e in examples], ["A", "B", "C"])
        self.assertTrue(all(e["inference_ready"] for e in examples))
        self.assertTrue(all(e["available_stages"] == ["pre_round", "post_first_kill"] for e in examples))

    def test_six_explicit_snapshots_are_stage_bound_and_spoiler_free(self):
        with patch.object(self.service, 'outcome', side_effect=AssertionError('No spoilers')):
            for example in ('A', 'B', 'C'):
                for stage, count in (('pre_round', 27), ('post_first_kill', 31)):
                    status, raw, _ = self.request('GET', f'/api/examples/{example}/snapshots/{stage}')
                    self.assertEqual(status, 200)
                    snapshot = json.loads(raw)
                    self.assertEqual(snapshot, self.service.snapshot(example, stage))
                    self.assertEqual(len(snapshot['features']), count)
                    self.assertNotIn(b'winning_side', raw)
                    self.assertNotIn(b'reference_probabilities', raw)
                    if stage == 'pre_round':
                        self.assertNotIn(b'first_kill', raw)
        for path in ('/api/examples/D/snapshots/pre_round', '/api/examples/A/snapshots/first_kill',
                     '/api/examples/A/snapshots/unknown'):
            self.assertEqual(self.request('GET', path)[0], 404)

    def test_all_twelve_http_predictions_match_trusted_references(self):
        reference = self.service._registry["reference_probabilities"]
        observed_request_ids = set()
        for example_id in ("A", "B", "C"):
            for (stage, algorithm), (model_id, model_path, calibrator_path) in MODEL_FILES.items():
                with self.subTest(example_id=example_id, stage=stage, algorithm=algorithm):
                    body = json.dumps({"example_id": example_id, "stage": stage, "algorithm": algorithm})
                    status, data, _ = self.request(
                        "POST", "/api/predict", body, {"Content-Type": "application/json"}
                    )
                    self.assertEqual(status, 200, data.decode("utf-8", errors="replace"))
                    result = json.loads(data)
                    actual = result["prediction"]["ct_win_probability"]
                    self.assertAlmostEqual(actual, reference[example_id][model_id], delta=1e-8)
                    self.assertAlmostEqual(
                        actual + result["prediction"]["t_win_probability"], 1.0, delta=1e-12
                    )
                    self.assertEqual(result["example_id"], example_id)
                    self.assertEqual(result["stage"], stage)
                    self.assertEqual(result["algorithm"], algorithm)
                    self.assertEqual(result["model_id"], model_id)
                    self.assertEqual(result["status"], "success")
                    self.assertEqual(
                        result["model_sha256"], self.service._registry["files"][model_path]["sha256"]
                    )
                    self.assertEqual(
                        result["calibrator_sha256"],
                        self.service._registry["files"][calibrator_path]["sha256"],
                    )
                    request_id = result["request_id"]
                    self.assertEqual(str(UUID(request_id)), request_id)
                    self.assertNotIn(request_id, observed_request_ids)
                    observed_request_ids.add(request_id)
        self.assertEqual(len(observed_request_ids), 12)

    def test_strict_json_and_request_shape(self):
        base = {"example_id": "A", "stage": "pre_round", "algorithm": "xgboost"}
        bodies = ["{", "null", "[]", "true", '{"example_id":"A","example_id":"B"}',
                  json.dumps({**base, "path": "C:/private"}), json.dumps({**base, "example_id": []}),
                  json.dumps({**base, "stage": "first_kill"}), json.dumps({**base, "example_id": "D"}),
                  json.dumps({**base, "algorithm": "random_forest"}),
                  '{"example_id":NaN,"stage":"pre_round","algorithm":"xgboost"}']
        for body in bodies:
            with self.subTest(body=body):
                self.assertEqual(self.request("POST", "/api/predict", body, {"Content-Type": "application/json"})[0], 400)
        self.assertEqual(self.request("POST", "/api/predict", "{}", {"Content-Type": "text/plain"})[0], 415)
        self.assertEqual(self.request("POST", "/api/predict", " " * 4097, {"Content-Type": "application/json"})[0], 413)

    def test_host_origin_and_fetch_site_rejected(self):
        for headers in ({"Host": "evil.example"}, {"Origin": "https://evil.example"},
                        {"Origin": "null"}, {"Sec-Fetch-Site": "cross-site"},
                        {"Host": "127.0.0.1:1"}):
            with self.subTest(headers=headers):
                self.assertEqual(self.request("GET", "/api/models", headers=headers)[0], 403)
        self.assertEqual(self.request("GET", "/api/models", headers={"Origin": f"http://127.0.0.1:{self.port}"})[0], 200)

    def test_no_repository_download_or_path_traversal(self):
        for path in ("/examples/roundcast_v1_cases.json", "/../examples/roundcast_v1_cases.json",
                     "/%2e%2e/examples/roundcast_v1_cases.json", "/models/", "/app.js?path=../",
                     "/api/examples/A?stage=post_first_kill", "/api/metrics?path=../",
                     "/api/metrics/../../examples/roundcast_v1_cases.json", "/api/unknown"):
            with self.subTest(path=path):
                self.assertIn(self.request("GET", path)[0], (400, 404))

    def test_ambiguous_headers_invalid_utf8_and_deep_json_never_run_models(self):
        authority = f'127.0.0.1:{self.port}'
        cases = [
            (f'Host: {authority}\r\nHost: {authority}\r\n', b'{}', 403),
            (f'Host: {authority}\r\nOrigin: http://{authority}\r\nOrigin: http://{authority}\r\n', b'{}', 403),
            (f'Host: {authority}\r\nContent-Length: 2\r\n', b'{}', 400),
            (f'Host: {authority}\r\nTransfer-Encoding: chunked\r\n', b'{}', 400),
            (f'Host: {authority}\r\nSec-Fetch-Site: same-site\r\n', b'{}', 403),
            (f'Host: {authority}\r\n', b'\xff', 400),
            (f'Host: {authority}\r\n', b'[' * 1500 + b']' * 1500, 400),
        ]
        with patch.object(self.service, 'predict_example') as predict:
            for headers, body, expected in cases:
                with self.subTest(headers=headers, size=len(body)), socket.create_connection(('127.0.0.1', self.port), timeout=5) as sock:
                    request = f'POST /api/predict HTTP/1.1\r\n{headers}Content-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n'.encode() + body
                    sock.sendall(request)
                    self.assertEqual(int(sock.recv(4096).split(b' ')[1]), expected)
            with socket.create_connection(('127.0.0.1', self.port), timeout=5) as sock:
                sock.sendall(f'POST /api/predict HTTP/1.1\r\nHost: {authority}\r\nContent-Type: application/json\r\n\r\n'.encode())
                self.assertEqual(int(sock.recv(4096).split(b' ')[1]), 400)
            predict.assert_not_called()

    def test_nested_diagnostic_metadata_never_reaches_http(self):
        from src.csdemo.predict_pre_round import PreRoundPredictor
        original = PreRoundPredictor.predict
        def malformed(predictor, snapshot):
            result = original(predictor, snapshot)
            result['validation']['private_path'] = 'C:/SYNTHETIC_PRIVATE/diagnostic.txt'
            return result
        with patch.object(PreRoundPredictor, 'predict', autospec=True, side_effect=malformed):
            status, raw, _ = self.request('POST', '/api/predict',
                json.dumps({'example_id': 'A', 'stage': 'pre_round', 'algorithm': 'xgboost'}), {'Content-Type': 'application/json'})
        self.assertEqual(status, 503)
        self.assertEqual(set(json.loads(raw)), {'status', 'message'})
        self.assertNotIn(b'SYNTHETIC_PRIVATE', raw)

    def test_each_model_chain_failure_is_sanitized_and_never_falls_back(self):
        with patch.object(self.service, "predict_example", side_effect=RuntimeError("C:\\private\\auth.json traceback")):
            for stage, algorithm in MODEL_FILES:
                with self.subTest(stage=stage, algorithm=algorithm):
                    body = json.dumps({"example_id": "A", "stage": stage, "algorithm": algorithm})
                    status, data, _ = self.request(
                        "POST", "/api/predict", body, {"Content-Type": "application/json"}
                    )
                    self.assertEqual(status, 503)
                    response = json.loads(data)
                    self.assertEqual(set(response), {"status", "message"})
                    self.assertEqual(response["status"], "error")
                    self.assertIn("未使用备用概率", response["message"])
                    for leaked in (b"private", b"auth.json", b"traceback", b"prediction", b"reference"):
                        self.assertNotIn(leaked, data)

    def test_local_only_binding_and_response_security(self):
        with self.assertRaises(ValueError):
            create_server(host="0.0.0.0", port=0, service=self.service)
        status, _, headers = self.request("GET", "/api/models")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
        self.assertNotIn("Access-Control-Allow-Origin", headers)
        self.assertEqual(self.request("PUT", "/api/models")[0], 405)

    def test_huge_length_gets_http_error_without_loading_model(self):
        with patch.object(self.service, "predict_example", side_effect=AssertionError("unsafe call")) as predict:
            with socket.create_connection(("127.0.0.1", self.port), timeout=5) as sock:
                raw = f"POST /api/predict HTTP/1.1\r\nHost: 127.0.0.1:{self.port}\r\nContent-Type: application/json\r\nContent-Length: " + "9" * 5000 + "\r\n\r\n"
                sock.sendall(raw.encode("ascii"))
                reply = sock.recv(4096)
                self.assertIn(b" 413 ", reply)
            predict.assert_not_called()


if __name__ == "__main__":
    unittest.main()
