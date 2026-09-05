"""Loopback-only ROUNDCAST demonstration server; no arbitrary file access."""
from __future__ import annotations

import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .roundcast_service import MODEL_FILES, RoundcastService, RoundcastValidationError, _unique_object
from .roundcast_chat import ChatManager, ChatError

STATIC_FILES = {"/": ("index.html", "text/html; charset=utf-8"),
                "/index.html": ("index.html", "text/html; charset=utf-8"),
                "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                "/chat.js": ("chat.js", "text/javascript; charset=utf-8"),
                "/styles.css": ("styles.css", "text/css; charset=utf-8")}


def _reject_constant(value):
    raise ValueError("Non-finite JSON constant")


def create_server(host: str = "127.0.0.1", port: int = 8765,
                  service: RoundcastService | None = None, chat=None) -> ThreadingHTTPServer:
    if host != "127.0.0.1":
        raise ValueError("Only 127.0.0.1 is allowed")
    trusted = service or RoundcastService()
    dialogue = chat or ChatManager(trusted)
    web_root = Path(__file__).resolve().parents[2] / "web/roundcast"

    class Handler(BaseHTTPRequestHandler):
        server_version = "ROUNDCAST"
        sys_version = ""

        def log_message(self, format, *args):
            pass

        def reply(self, status, payload, content_type="application/json; charset=utf-8"):
            body = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
            self.end_headers()
            self.wfile.write(body)

        def error(self, status, message):
            self.reply(status, {"status": "error", "message": message})

        def do_GET(self):
            if not self.allowed_request():
                return
            if self.path == "/api/chat/status":
                return self.reply(200, dialogue.status())
            if re.fullmatch(r"/api/chat/jobs/[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", self.path):
                try:
                    return self.reply(200, dialogue.job(self.path.rsplit("/", 1)[1]))
                except KeyError:
                    return self.error(404, "对话请求已过期或不存在，请重新发送")
            if self.path == "/api/models":
                return self.reply(200, {"models": [trusted.model_metadata(*pair) for pair in MODEL_FILES]})
            if self.path == "/api/metrics":
                return self.reply(200, trusted.metrics())
            if self.path == "/api/examples":
                return self.reply(200, {"examples": [{**e, "inference_ready": True,
                                                      "available_stages": ["pre_round", "post_first_kill"]}
                                                     for e in trusted.examples()]})
            for e in ("A", "B", "C"):
                for stage in ("pre_round", "post_first_kill"):
                    if self.path == f"/api/examples/{e}/snapshots/{stage}":
                        return self.reply(200, trusted.snapshot(e, stage))
                if self.path == f"/api/examples/{e}":
                    return self.reply(200, trusted.snapshot(e, "pre_round"))
                if self.path == f"/api/examples/{e}/outcome":
                    return self.reply(200, {"example_id": e, "winning_side": trusted.outcome(e)["winning_side"]})
            if self.path in STATIC_FILES:
                name, content_type = STATIC_FILES[self.path]
                try:
                    return self.reply(200, (web_root / name).read_bytes(), content_type)
                except OSError:
                    return self.error(404, "页面资源尚不可用")
            self.error(404, "未找到此资源")

        def allowed_request(self):
            authority = f"127.0.0.1:{self.server.server_address[1]}"
            hosts = self.headers.get_all("Host", [])
            origins = self.headers.get_all("Origin", [])
            if (hosts != [authority] or len(origins) > 1
                    or (origins and origins != [f"http://{authority}"])
                    or self.headers.get("Sec-Fetch-Site") in ("cross-site", "same-site")):
                self.error(403, "只允许从本机同源页面访问")
                return False
            if "?" in self.path or "#" in self.path or "%" in self.path or "\\" in self.path or ".." in self.path:
                self.error(400, "请求路径无效")
                return False
            return True

        def do_POST(self):
            if not self.allowed_request():
                return
            if self.path not in ("/api/predict", "/api/chat", "/api/chat/cancel"):
                return self.error(404, "未找到此接口")
            if self.headers.get_all("Transfer-Encoding"):
                return self.error(400, "不支持分块请求")
            lengths = self.headers.get_all("Content-Length", [])
            if len(lengths) != 1 or not lengths[0].isascii() or not lengths[0].isdigit():
                return self.error(400, "请求长度无效")
            if len(lengths[0]) > 10:
                return self.error(413, "请求内容过大")
            size = int(lengths[0])
            if size > (131072 if self.path == "/api/chat" else 4096):
                return self.error(413, "请求内容过大")
            if self.headers.get("Content-Type", "").split(";")[0].strip().lower() != "application/json":
                return self.error(415, "仅支持 JSON 请求")
            try:
                self.connection.settimeout(5)
                raw = self.rfile.read(size)
                if len(raw) != size:
                    raise ValueError("Incomplete body")
                body = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object,
                                  parse_constant=_reject_constant)
            except (ValueError, OSError, RecursionError):
                return self.error(400, "JSON 格式无效")
            if self.path in ("/api/chat", "/api/chat/cancel"):
                try:
                    if self.path == "/api/chat":
                        return self.reply(202, dialogue.start(body))
                    if (not isinstance(body, dict) or set(body) != {"job_id"}
                            or not isinstance(body['job_id'], str)):
                        raise ValueError("Invalid job")
                    return self.reply(200, dialogue.cancel(body['job_id']))
                except ValueError:
                    return self.error(400, "问题、历史记录或回合信息无效；预测过期时请重新运行模型")
                except KeyError:
                    return self.error(404, "对话请求已过期或不存在")
                except ChatError as exc:
                    return self.error(429 if exc.code == 'busy' else 503, exc.message)
            try:
                if (not isinstance(body, dict) or set(body) != {"example_id", "stage", "algorithm"}
                        or any(not isinstance(value, str) for value in body.values())):
                    raise ValueError("Invalid fields")
                trusted.snapshot(body["example_id"], body["stage"])
                model = trusted.model_metadata(body["stage"], body["algorithm"])
            except (ValueError, OSError, RecursionError):
                return self.error(400, "案例、时点、算法或 JSON 格式无效")
            if body["example_id"] not in model["available_examples"]:
                return self.error(409, "此案例与模型组合尚未开放")
            try:
                result = trusted.predict_example(**body)
                dialogue.remember_prediction(result)
                self.reply(200, result)
            except Exception:
                self.error(503, "模型或可信来源不可用；未使用备用概率，请检查本地文件后重试")

        def unsupported(self):
            self.error(405, "不支持此请求方法")

        do_PUT = do_DELETE = do_PATCH = do_OPTIONS = do_HEAD = unsupported

    class LocalServer(ThreadingHTTPServer):
        def server_close(self):
            dialogue.close()
            super().server_close()

    return LocalServer((host, port), Handler)


def main():
    parser = argparse.ArgumentParser(description="ROUNDCAST local demonstration")
    parser.add_argument("--host", default="127.0.0.1", choices=["127.0.0.1"])
    parser.add_argument("--port", default=8765, type=int)
    args = parser.parse_args()
    try:
        server = create_server(args.host, args.port)
    except (ValueError, OSError):
        parser.exit(2, "Cannot start: verify trusted local artifacts and port availability.\n")
    print(f"ROUNDCAST ready: http://127.0.0.1:{server.server_address[1]}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
