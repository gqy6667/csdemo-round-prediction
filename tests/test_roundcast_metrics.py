import http.client
import json
import threading
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.csdemo.roundcast_service import RoundcastService
from src.csdemo.roundcast_server import create_server


class RoundcastMetricsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = RoundcastService()

    def test_four_formal_sources_match_exactly_without_prediction_or_case_labels(self):
        with patch('joblib.load', side_effect=AssertionError('Metrics must not load models')):
            result = self.service.metrics()
        self.assertEqual(result['status'], 'success')
        self.assertEqual(len(result['models']), 4)
        expected_sources = {
            'xgb_pre_round': ('reports/esta_full_m10/m10_summary.json', .7271219343412749),
            'xgb_post_first_kill': ('reports/esta_full_m18/m18_summary.json', .8098368635821109),
            'lgbm_pre_round': ('reports/esta_full_m27/fixed_test_metrics.csv', .7278463079406734),
            'lgbm_post_first_kill': ('reports/esta_full_m33/fixed_test_metrics.csv', .808255446182),
        }
        for row in result['models']:
            self.assertEqual(row['source']['path'], expected_sources[row['model_id']][0])
            self.assertAlmostEqual(row['metrics']['auc'], expected_sources[row['model_id']][1], places=14)
            self.assertEqual(row['status'], 'success')
            self.assertEqual(row['scope'], 'frozen_test_set')
            self.assertEqual(row['n_test'], 4172 if row['stage'] == 'pre_round' else 4170)
            path = self.service.root / row['source']['path']
            if row['algorithm'] == 'lightgbm':
                expected = pd.read_csv(path).set_index('metric')['value'].to_dict()
            else:
                document = json.loads(path.read_text(encoding='utf-8'))
                expected = document['test_selected_metrics'] if row['stage'] == 'pre_round' else document['calibration']['selected_test_metrics']
            self.assertEqual(set(row['metrics']), {'accuracy', 'auc', 'log_loss', 'brier_score', 'ece10'})
            for name, value in expected.items():
                self.assertAlmostEqual(row['metrics'][name], value, places=14)
        for forbidden in ('winning_side', 'example_id', 'reference_probabilities'):
            self.assertNotIn(forbidden, json.dumps(result))
        self.assertNotIn(str(self.service.root).replace('\\', '/'), json.dumps(result).replace('\\\\', '/'))
        self.service.snapshot('B', 'post_first_kill')
        self.assertEqual(result, self.service.metrics())

    def test_missing_or_tampered_metric_source_only_disables_that_model_and_can_recover(self):
        original = Path.read_bytes
        for mode in ('missing', 'changed'):
            def fail(path):
                if path.name == 'fixed_test_metrics.csv' and path.parent.name == 'esta_full_m33':
                    if mode == 'missing':
                        raise FileNotFoundError('private diagnostic')
                    return original(path) + b'tampered'
                return original(path)
            with self.subTest(mode=mode), patch.object(Path, 'read_bytes', fail):
                report = self.service.metrics()
                self.assertEqual(report['status'], 'partial')
                rows = {row['model_id']: row for row in report['models']}
                self.assertIsNone(rows['lgbm_post_first_kill']['metrics'])
                self.assertEqual(rows['lgbm_post_first_kill']['status'], 'unavailable')
                self.assertEqual(sum(row['status'] == 'success' for row in rows.values()), 3)
                self.assertNotIn('private diagnostic', json.dumps(report))
                self.assertEqual(self.service.snapshot('A', 'pre_round')['example_id'], 'A')
                if mode == 'missing':
                    self.assertEqual(self.service.predict_example('A', 'post_first_kill', 'lightgbm')['status'], 'success')
        self.assertEqual(self.service.metrics()['status'], 'success')

    def test_invalid_metrics_are_rejected_without_zero_fallback(self):
        from src.csdemo.roundcast_service import parse_formal_metrics
        valid = {'accuracy': .7, 'auc': .8, 'log_loss': .5, 'brier_score': .2, 'ece10': .01}
        edge = {**valid, 'accuracy': 0, 'ece10': 0, 'log_loss': 2}
        self.assertEqual(parse_formal_metrics(json.dumps({'selected': edge}).encode(), ('selected',)), edge)
        for change in ({'auc': None}, {'accuracy': True}, {'ece10': float('nan')}, {'brier_score': 2}, {'log_loss': -1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                parse_formal_metrics(json.dumps({'test_selected_metrics': {**valid, **change}}).encode(), ('test_selected_metrics',))
        missing = deepcopy(valid)
        del missing['auc']
        with self.assertRaises(ValueError):
            parse_formal_metrics(json.dumps({'test_selected_metrics': missing}).encode(), ('test_selected_metrics',))
        with self.assertRaises(ValueError):
            parse_formal_metrics(b'{"selected": {}, "selected": {}}', ('selected',))
        for raw in (b'metric,value\naccuracy,0.7\naccuracy,0.8\n', b'metric,value\nauc,NaN\n'):
            with self.assertRaises(ValueError):
                parse_formal_metrics(raw, ())

    def test_extra_csv_fields_and_numeric_overflow_return_validation_errors(self):
        from src.csdemo.roundcast_service import parse_formal_metrics, RoundcastValidationError
        raw = b'metric,value\nextra,accuracy,0.7\nextra,auc,0.8\nextra,log_loss,0.5\nextra,brier_score,0.2\nextra,ece10,0.01\n'
        with self.assertRaises(RoundcastValidationError):
            parse_formal_metrics(raw, ())
        huge = {'accuracy': .7, 'auc': .8, 'log_loss': 10**400, 'brier_score': .2, 'ece10': .01}
        with self.assertRaises(RoundcastValidationError):
            parse_formal_metrics(json.dumps({'selected': huge}).encode(), ('selected',))

    def test_metrics_http_contract_and_request_guards(self):
        server = create_server(port=0, service=self.service)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            def request(path, headers=None):
                connection = http.client.HTTPConnection('127.0.0.1', server.server_address[1], timeout=10)
                connection.request('GET', path, headers=headers or {})
                response = connection.getresponse()
                status, body = response.status, response.read()
                connection.close()
                return status, body
            status, body = request('/api/metrics')
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body), self.service.metrics())
            self.assertEqual(request('/api/metrics?example_id=A')[0], 400)
            self.assertEqual(request('/api/metrics', {'Origin': 'https://invalid.example'})[0], 403)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(5)

    def test_all_metric_and_sample_file_failures_are_isolated(self):
        from src.csdemo.roundcast_service import METRIC_SOURCES
        original = Path.read_bytes
        for model, source in METRIC_SOURCES.items():
            for relative in (source[0], self.service._registry['reference_sources'][model]['path']):
                for mode in ('missing', 'tampered'):
                    def altered(path):
                        if path == self.service.root / relative:
                            if mode == 'missing':
                                raise FileNotFoundError('synthetic-private-path')
                            return original(path) + b'tampered'
                        return original(path)
                    with self.subTest(model=model, source=relative, mode=mode), patch.object(Path, 'read_bytes', altered):
                        report = self.service.metrics()
                        self.assertEqual(report['status'], 'partial')
                        rows = {row['model_id']: row for row in report['models']}
                        self.assertEqual(rows[model]['status'], 'unavailable')
                        self.assertIsNone(rows[model]['metrics'])
                        self.assertEqual(sum(row['status'] == 'success' for row in rows.values()), 3)
                        self.assertNotIn('synthetic-private-path', json.dumps(report))
        self.assertEqual(self.service.metrics()['status'], 'success')
