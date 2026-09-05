"""T10 cross-layer acceptance. Faults are in-memory; never modify frozen files."""
import hashlib
import http.client
import json
import threading
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.csdemo.roundcast_server import create_server
from src.csdemo.roundcast_service import DATA_FILES, KEY_FIELDS, METRIC_SOURCES, MODEL_FILES, RoundcastService


def main():
    service = RoundcastService()
    registry = service._registry
    pins = {p: v['sha256'] for p, v in registry['files'].items()}
    pins.update(registry['manifest_pins'])
    pins.update({source[0]: source[1] for source in METRIC_SOURCES.values()})
    def hashes():
        return {p: hashlib.sha256((service.root / p).read_bytes()).hexdigest() for p in pins}
    before = hashes()
    assert before == pins
    references = {model: pd.read_csv(BytesIO(service._check_file(source['path'], pins[source['path']])),
                                   float_precision='round_trip').set_index(list(KEY_FIELDS))
                  for model, source in registry['reference_sources'].items()}
    server = create_server(port=0, service=service)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    def request(method, path, body=None, headers=None):
        connection = http.client.HTTPConnection('127.0.0.1', server.server_address[1], timeout=10)
        try:
            connection.request(method, path, json.dumps(body) if body is not None else None,
                               headers if headers is not None else {'Content-Type': 'application/json'})
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()
    records, snapshots, failures, guards = [], [], [], []
    try:
        assert len(request('GET', '/api/models')[1]['models']) == 4
        assert len(request('GET', '/api/examples')[1]['examples']) == 3
        metrics = request('GET', '/api/metrics')[1]
        assert metrics['status'] == 'success'
        for case in registry['cases']:
            example = case['example_id']
            for stage in DATA_FILES:
                status, detail = request('GET', f'/api/examples/{example}/snapshots/{stage}')
                assert status == 200 and detail == service.snapshot(example, stage)
                assert not {'ct_win', 'winning_side', 'reference_probabilities'} & detail.keys()
                snapshots.append({'example_id': example, 'stage': stage, 'input_count': len(detail['features'])})
            for (stage, algorithm), (model, _, _) in MODEL_FILES.items():
                body = {'example_id': example, 'stage': stage, 'algorithm': algorithm}
                status, result = request('POST', '/api/predict', body)
                assert status == 200 and result['identity'] == case['identity']
                source = registry['reference_sources'][model]
                reference = float(references[model].loc[tuple(case['identity'][k] for k in KEY_FIELDS), source['probability_column']])
                error = abs(result['prediction']['ct_win_probability'] - reference)
                assert error <= 1e-8 and all(result[k] == v for k, v in body.items())
                records.append({'request': body, 'response': result, 'reference': reference, 'absolute_error': error})
            assert request('GET', f'/api/examples/{example}/outcome')[1]['winning_side'] == service.outcome(example)['winning_side']
        # Missing/tampered active artifacts must fail before deserialization.
        original = Path.read_bytes
        for (stage, algorithm), (_, model_path, calibration_path) in MODEL_FILES.items():
            for relative in (model_path, calibration_path, DATA_FILES[stage]):
                for mode in ('missing', 'tampered'):
                    def altered(path):
                        if path == service.root / relative:
                            if mode == 'missing':
                                raise FileNotFoundError('synthetic-private-diagnostic')
                            return original(path) + b'tampered'
                        return original(path)
                    with patch.object(Path, 'read_bytes', altered), patch('joblib.load') as load:
                        status, body = request('POST', '/api/predict', {'example_id': 'A', 'stage': stage, 'algorithm': algorithm})
                        assert status == 503 and set(body) == {'status', 'message'}
                        assert 'synthetic-private-diagnostic' not in json.dumps(body)
                        load.assert_not_called()
                    failures.append({'stage': stage, 'algorithm': algorithm, 'source': relative, 'fault': mode, 'http_status': status, 'model_loads': 0})
        for path, headers, expected in [('/api/metrics?path=../', {}, 400), ('/models/', {}, 404),
                ('/examples/roundcast_v1_cases.json', {}, 404), ('/api/models', {'Host': 'untrusted.invalid'}, 403),
                ('/api/metrics', {'Origin': 'https://untrusted.invalid'}, 403)]:
            assert request('GET', path, headers=headers)[0] == expected
            guards.append({'path': path, 'expected_status': expected})
    finally:
        server.shutdown()
        server.server_close()
        worker.join(5)
    after = hashes()
    assert before == after == pins and len(records) == 12
    assert len({r['response']['request_id'] for r in records}) == 12
    evidence = {'status': 'passed', 'recorded_at': datetime.now(timezone.utc).isoformat(), 'scope': 'T10 local cross-layer acceptance',
                'api_categories': ['models', 'examples', 'snapshots', 'predict', 'outcome', 'metrics'],
                'records': records, 'snapshots': snapshots, 'metrics': metrics, 'faults': failures, 'guards': guards,
                'hashes_before': before, 'hashes_after': after, 'frozen_sources_unchanged': True,
                'max_absolute_error': max(row['absolute_error'] for row in records)}
    (service.root / 'reports/roundcast_interactive_v1/t10-api-evidence.json').write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'passed', 'real_predictions': len(records), 'faults': len(failures),
                      'pinned_files': len(pins), 'max_absolute_error': evidence['max_absolute_error']}))


if __name__ == '__main__':
    main()
