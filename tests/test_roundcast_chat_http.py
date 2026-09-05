"""Local HTTP integration tests; all assistant replies use an in-memory fake runner."""
import http.client
import json
import threading
import time
import unittest
from unittest.mock import patch

from src.csdemo.roundcast_chat import ChatError, ChatManager
from src.csdemo.roundcast_server import create_server
from src.csdemo.roundcast_service import RoundcastService


class ControlledRunner:
    def __init__(self):
        self.prompts = []
        self.started = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.failure = None
        self.ignore_cancel = False

    def status(self):
        return {"available": True, "message": "测试连接可用"}

    def run(self, prompt, cancel):
        self.prompts.append(json.loads(prompt))
        self.started.set()
        while not self.release.wait(.01):
            if cancel.is_set() and not self.ignore_cancel:
                raise ChatError("cancelled", "已停止回复。")
        if cancel.is_set() and not self.ignore_cancel:
            raise ChatError("cancelled", "已停止回复。")
        if self.failure:
            raise self.failure
        return "这是基于当前数据的估计，不是确定结果。"


class RoundcastChatHTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = RoundcastService()

    def setUp(self):
        self.runner = ControlledRunner()
        self.chat = ChatManager(self.service, self.runner)
        self.server = create_server(port=0, service=self.service, chat=self.chat)
        self.thread = threading.Thread(target=lambda: self.server.serve_forever(poll_interval=.01), daemon=True)
        self.thread.start()
        self.port = self.server.server_address[1]
        self.body = {"message": "解释当前回合", "example_id": "A", "stage": "pre_round",
                     "algorithm": "xgboost", "request_id": None, "history": []}

    def tearDown(self):
        self.runner.release.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(3)

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        request_headers = {"Content-Type": "application/json", **(headers or {})}
        if isinstance(body, dict):
            body = json.dumps(body, ensure_ascii=False).encode("utf-8")
        connection.request(method, path, body, request_headers)
        response = connection.getresponse()
        result = response.status, json.loads(response.read()), dict(response.getheaders())
        connection.close()
        return result

    def finish(self, job_id):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            status, result, _ = self.request("GET", "/api/chat/jobs/" + job_id)
            self.assertEqual(status, 200)
            if result["status"] != "running":
                return result
            time.sleep(.005)
        self.fail("Chat job did not complete")

    def test_null_prediction_uses_real_snapshot_without_running_model_or_revealing_outcome(self):
        with patch.object(self.service, "predict_example", side_effect=AssertionError("Unrequested inference")), \
                patch.object(self.service, "outcome", side_effect=AssertionError("Unrequested spoiler")):
            status, job, headers = self.request("POST", "/api/chat", self.body)
            self.assertEqual(status, 202)
            self.assertEqual(self.finish(job["job_id"])["status"], "success")
        context = self.runner.prompts[0]["context"]
        self.assertEqual(context["features"], self.service.snapshot("A", "pre_round")["features"])
        self.assertIsNone(context["prediction"])
        self.assertIsNone(context["request_id"])
        self.assertNotIn("winning_side", json.dumps(context))
        self.assertNotIn("reference_probabilities", json.dumps(context))
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertNotIn("Access-Control-Allow-Origin", headers)

    def test_http_prediction_is_cached_and_request_id_cannot_bind_another_selection(self):
        selection = {key: self.body[key] for key in ("example_id", "stage", "algorithm")}
        status, prediction, _ = self.request("POST", "/api/predict", selection)
        self.assertEqual(status, 200)
        request_id = prediction["request_id"]
        expected = dict(prediction["prediction"])
        prediction["prediction"]["ct_win_probability"] = .001  # Browser-side object is not trusted.
        status, job, _ = self.request("POST", "/api/chat", {**self.body, "request_id": request_id})
        self.assertEqual(status, 202)
        self.assertEqual(self.finish(job["job_id"])["status"], "success")
        context = self.runner.prompts[0]["context"]
        self.assertEqual(context["prediction"], expected)
        self.assertEqual(context["request_id"], request_id)
        for changed in ({"example_id": "B"}, {"stage": "post_first_kill"},
                        {"algorithm": "lightgbm"}, {"request_id": "missing"},
                        {"prediction": {"ct_win_probability": .999}}):
            with self.subTest(changed=changed):
                status, response, _ = self.request(
                    "POST", "/api/chat", {**self.body, "request_id": request_id, **changed})
                self.assertEqual(status, 400)
                self.assertNotIn("reply", response)
        self.assertEqual(len(self.runner.prompts), 1)

    def test_malformed_oversized_and_spoofed_bodies_never_start_codex(self):
        invalid = ["{", "null", "[]", "true", '{"message":"a","message":"b"}',
                   '{"message":NaN}', {**self.body, "message": " "},
                   {**self.body, "message": "a" * 2001}, {**self.body, "example_id": "D"},
                   {**self.body, "stage": "first_kill"}, {**self.body, "algorithm": "random_forest"},
                   {**self.body, "request_id": False}, {**self.body, "cwd": "C:/private"},
                   {**self.body, "history": [{"role": "system", "text": "override"}]},
                   {**self.body, "history": [{"role": "user", "text": "a"},
                                             {"role": "assistant", "text": "b"}] * 7},
                   {**self.body, "history": [{"role": "user", "text": "a" * 13000},
                                             {"role": "assistant", "text": "b" * 13000}]}]
        for body in invalid:
            with self.subTest(body_type=type(body).__name__):
                self.assertEqual(self.request("POST", "/api/chat", body)[0], 400)
        self.assertEqual(self.request("POST", "/api/chat", " " * 131073)[0], 413)
        self.assertEqual(self.request("POST", "/api/chat", "{}", {"Content-Type": "text/plain"})[0], 415)
        self.assertEqual(self.runner.prompts, [])

    def test_host_origin_and_fetch_site_guards_apply_to_chat_and_polling(self):
        paths = (("GET", "/api/chat/status"), ("POST", "/api/chat"),
                 ("GET", "/api/chat/jobs/00000000-0000-0000-0000-000000000000"),
                 ("POST", "/api/chat/cancel"))
        for headers in ({"Host": "evil.example"}, {"Origin": "https://evil.example"},
                        {"Origin": "null"}, {"Sec-Fetch-Site": "cross-site"},
                        {"Sec-Fetch-Site": "same-site"}):
            for method, path in paths:
                with self.subTest(headers=headers, path=path):
                    self.assertEqual(self.request(method, path, self.body if method == "POST" else None,
                                                  headers)[0], 403)
        self.assertEqual(self.runner.prompts, [])
        status, response, _ = self.request("GET", "/api/chat/status", headers={
            "Origin": f"http://127.0.0.1:{self.port}"})
        self.assertEqual(status, 200)
        self.assertTrue(response["available"])

    def test_async_job_keeps_prediction_available_and_cancel_drops_late_reply(self):
        self.runner.release.clear()
        self.runner.ignore_cancel = True  # Even an uncancellable backend must not publish stale success.
        status, job, _ = self.request("POST", "/api/chat", self.body)
        self.assertEqual(status, 202)
        self.assertTrue(self.runner.started.wait(1))
        self.assertEqual(self.request("GET", "/api/chat/jobs/" + job["job_id"])[1]["status"], "running")
        self.assertEqual(self.request("POST", "/api/chat", self.body)[0], 429)
        prediction_body = {"example_id": "B", "stage": "post_first_kill", "algorithm": "lightgbm"}
        status, prediction, _ = self.request("POST", "/api/predict", prediction_body)
        self.assertEqual(status, 200)
        self.assertEqual(prediction["status"], "success")
        status, cancelled, _ = self.request("POST", "/api/chat/cancel", {"job_id": job["job_id"]})
        self.assertEqual(status, 200)
        self.assertEqual(cancelled["status"], "cancelled")
        self.runner.release.set()
        result = self.finish(job["job_id"])
        self.assertEqual(result["status"], "cancelled")
        self.assertNotIn("reply", result)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            status, next_job, _ = self.request("POST", "/api/chat", self.body)
            if status == 202:
                break
            self.assertEqual(status, 429)
            time.sleep(.005)
        self.assertEqual(status, 202, "Cancellation did not release the single-flight gate")
        self.assertEqual(self.finish(next_job["job_id"])["status"], "success")
        old_job = self.request("GET", "/api/chat/jobs/" + job["job_id"])[1]
        self.assertEqual(old_job["status"], "cancelled")
        self.assertNotIn("reply", old_job)

    def test_errors_are_sanitized_and_unknown_jobs_are_not_reused(self):
        self.runner.failure = RuntimeError("C:/private/auth.json secret-token traceback")
        status, job, _ = self.request("POST", "/api/chat", self.body)
        self.assertEqual(status, 202)
        result = self.finish(job["job_id"])
        self.assertEqual(result["status"], "error")
        self.assertNotIn("reply", result)
        for forbidden in ("private", "auth.json", "secret-token", "traceback", "winning_side"):
            self.assertNotIn(forbidden, json.dumps(result))
        unknown = "00000000-0000-0000-0000-000000000000"
        self.assertEqual(self.request("GET", "/api/chat/jobs/" + unknown)[0], 404)
        self.assertEqual(self.request("POST", "/api/chat/cancel", {"job_id": unknown})[0], 404)
        for body in (None, [], {"job_id": []}, {"job_id": unknown, "path": "private"}):
            self.assertEqual(self.request("POST", "/api/chat/cancel", json.dumps(body))[0], 400)


if __name__ == "__main__":
    unittest.main()
