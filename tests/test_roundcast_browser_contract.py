"""Static entry contract plus real JavaScript state tests (not browser E2E)."""
import os
from pathlib import Path
import shutil
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]


class RoundcastBrowserContractTests(unittest.TestCase):
    def test_technical_view_and_independent_metrics_contract(self):
        html = (ROOT / 'web/roundcast/index.html').read_text(encoding='utf-8')
        for item in ('id="technical-view"', 'data-view="technical"', 'id="technical-metrics"',
                     'id="metrics-reload"', 'id="technical-reveal"', 'id="technical-correctness"'):
            self.assertIn(item, html)
        bundled = Path.home() / '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe'
        node = os.environ.get('ROUNDCAST_NODE') or shutil.which('node') or str(bundled)
        result = subprocess.run([node, '--test', str(ROOT / 'tests/roundcast_technical.test.cjs')],
                                cwd=ROOT, capture_output=True, text=True, encoding='utf-8', timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_analysis_view_and_state_contract(self):
        html = (ROOT / 'web/roundcast/index.html').read_text(encoding='utf-8')
        for item in ('id="viewer-view"', 'id="analyst-view"', 'data-view="analyst"',
                     'id="analysis-chart"', 'id="analysis-snapshot"', 'id="analysis-economy"'):
            self.assertIn(item, html)
        bundled = Path.home() / '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe'
        node = os.environ.get('ROUNDCAST_NODE') or shutil.which('node') or str(bundled)
        result = subprocess.run([node, '--test', str(ROOT / 'tests/roundcast_analysis.test.cjs')],
                                cwd=ROOT, capture_output=True, text=True, encoding='utf-8', timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_analysis_field_allowlist_matches_actual_model_inputs(self):
        import json
        from src.csdemo.predict_pre_round import BASE_FEATURES
        from src.csdemo.m16_first_kill_baselines import FIRST_KILL_MODEL_FEATURES
        bundled = Path.home() / '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe'
        node = os.environ.get('ROUNDCAST_NODE') or shutil.which('node') or str(bundled)
        result = subprocess.run([node, '-e', "console.log(JSON.stringify(require('./web/roundcast/app.js').INPUT_FIELDS.map(f=>f[0])))"],
                                cwd=ROOT, capture_output=True, text=True, encoding='utf-8', timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        fields = json.loads(result.stdout)
        self.assertEqual(len(fields), 31)
        self.assertEqual(set(fields), set(BASE_FEATURES) | set(FIRST_KILL_MODEL_FEATURES))

    def test_selection_controls_and_four_result_comparison_exist(self):
        html = (ROOT / 'web/roundcast/index.html').read_text(encoding='utf-8')
        for item in ('id="run-all"', 'id="comparison-body"', 'id="first-kill-panel"',
                     'value="B"', 'value="C"', 'value="lightgbm"', 'value="post_first_kill"'):
            self.assertIn(item, html)

    def test_selection_state_contract(self):
        bundled = Path.home() / '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe'
        node = os.environ.get('ROUNDCAST_NODE') or shutil.which('node') or str(bundled)
        result = subprocess.run([node, '--test', str(ROOT / 'tests/roundcast_selection.test.cjs')],
                                cwd=ROOT, capture_output=True, text=True, encoding='utf-8', timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

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
