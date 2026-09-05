"""Static entry contract plus real JavaScript state tests (not browser E2E)."""
import os
from pathlib import Path
import shutil
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]


class RoundcastBrowserContractTests(unittest.TestCase):
    def test_static_entry_is_local_and_spoiler_free(self):
        html = (ROOT / "web/roundcast/index.html").read_text(encoding="utf-8")
        script = (ROOT / "web/roundcast/app.js").read_text(encoding="utf-8")
        css = (ROOT / "web/roundcast/styles.css").read_text(encoding="utf-8")
        for forbidden in ("教师", "老师", "0.629152", "62.92", "https://", "<iframe"):
            self.assertNotIn(forbidden, html + script)
        self.assertIn('src="/app.js"', html)
        self.assertIn('href="/styles.css"', html)
        self.assertIn('aria-live="polite"', html)
        self.assertNotIn("innerHTML", script)
        self.assertIn("@media", css)

    def test_javascript_state_contract(self):
        bundled = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe"
        node = os.environ.get("ROUNDCAST_NODE") or shutil.which("node") or str(bundled)
        result = subprocess.run([node, "--test", str(ROOT / "tests/roundcast_state.test.cjs")],
                                cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_javascript_chat_contract(self):
        html = (ROOT / "web/roundcast/index.html").read_text(encoding="utf-8")
        chat = (ROOT / "web/roundcast/chat.js").read_text(encoding="utf-8")
        self.assertLess(html.index('src="/chat.js"'), html.index('src="/app.js"'))
        for forbidden in ("innerHTML", "localStorage", "sessionStorage", "auth.json", "api_key"):
            self.assertNotIn(forbidden, chat)
        self.assertIn('id="chat-disclosure"', html)
        self.assertIn('maxlength="2000"', html)
        self.assertIn("不会发送隐藏赛果", html)
        bundled = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe"
        node = os.environ.get("ROUNDCAST_NODE") or shutil.which("node") or str(bundled)
        result = subprocess.run([node, "--test", str(ROOT / "tests/roundcast_chat.test.cjs")],
                                cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
