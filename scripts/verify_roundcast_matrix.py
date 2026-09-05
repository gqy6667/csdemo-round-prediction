"""Record twelve actual localhost predictions against pinned official CSV rows."""
from __future__ import annotations

import http.client
import json
import threading
from datetime import datetime, timezone
from io import BytesIO

import pandas as pd

from src.csdemo.roundcast_server import create_server
from src.csdemo.roundcast_service import KEY_FIELDS, MODEL_FILES, RoundcastService


def main() -> None:
    service = RoundcastService()
    registry = service._registry
    references = {}
    for model_id, source in registry["reference_sources"].items():
        content = service._check_file(source["path"], registry["files"][source["path"]]["sha256"])
        frame = pd.read_csv(BytesIO(content), float_precision="round_trip")
        if frame.duplicated(list(KEY_FIELDS)).any():
            raise ValueError("Duplicate reference identity")
        references[model_id] = frame.set_index(list(KEY_FIELDS))
    records = []
    server = create_server(port=0, service=service)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        for case in registry["cases"]:
            for (stage, algorithm), (model_id, _, _) in MODEL_FILES.items():
                request = {"example_id": case["example_id"], "stage": stage, "algorithm": algorithm}
                connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=10)
                try:
                    connection.request("POST", "/api/predict", json.dumps(request), {"Content-Type": "application/json"})
                    response = connection.getresponse()
                    result = json.loads(response.read())
                    if response.status != 200:
                        raise ValueError("Inference request failed")
                finally:
                    connection.close()
                source = registry["reference_sources"][model_id]
                reference = float(references[model_id].loc[tuple(case["identity"][k] for k in KEY_FIELDS), source["probability_column"]])
                error = abs(result["prediction"]["ct_win_probability"] - reference)
                if error > 1e-8 or any(result[k] != v for k, v in request.items()) or result["identity"] != case["identity"]:
                    raise ValueError("Prediction/reference identity or probability mismatch")
                records.append({"request": request, "http_status": 200, "response": result,
                                "reference_probability": reference, "absolute_error": error,
                                "reference_source": source,
                                "reference_source_sha256": registry["files"][source["path"]]["sha256"]})
    finally:
        server.shutdown()
        server.server_close()
        worker.join(5)
    if len(records) != 12 or len({r["response"]["request_id"] for r in records}) != 12:
        raise ValueError("Incomplete inference matrix")
    proof = service.readiness_report()
    original = json.loads((service.root / "reports/roundcast_interactive_v1/t01_readiness.json").read_text(encoding="utf-8"))
    unchanged = all(proof[k] == original[k] for k in ("frozen_artifacts", "supporting_files"))
    if not unchanged:
        raise ValueError("Trusted sources differ from T01")
    evidence = {"status": "passed", "recorded_at": datetime.now(timezone.utc).isoformat(),
                "scope": "T05: 12 real HTTP predictions, no frontend selection changes",
                "tolerance": 1e-8, "max_absolute_error": max(r["absolute_error"] for r in records),
                "frozen_sources_unchanged": unchanged, "records": records}
    output = service.root / "reports/roundcast_interactive_v1/t05_prediction_matrix.json"
    output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": evidence["status"], "count": len(records), "max_absolute_error": evidence["max_absolute_error"],
                      "probabilities": [{**r["request"], "ct": r["response"]["prediction"]["ct_win_probability"], "error": r["absolute_error"]} for r in records]}))


if __name__ == "__main__":
    main()
