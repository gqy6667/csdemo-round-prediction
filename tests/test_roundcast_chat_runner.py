"""Exercise subprocess boundaries without launching Codex or touching credentials."""
import json
import subprocess
import threading
import unittest
from unittest.mock import MagicMock, patch

from src.csdemo.roundcast_chat import ChatError, CodexRunner


def completed_output(text="当前尚未运行预测。"):
    return "\n".join(json.dumps(event, ensure_ascii=False) for event in (
        {"type": "item.completed", "item": {"type": "agent_message", "text": text}},
        {"type": "turn.completed"},
    )).encode("utf-8")


class CodexRunnerBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.runner = CodexRunner(executable="codex-test.exe")
        self.cancel = threading.Event()

    @staticmethod
    def child(output=None, returncode=0):
        process = MagicMock()
        process.communicate.return_value = (completed_output() if output is None else output, None)
        process.returncode = returncode
        process.poll.return_value = returncode
        return process

    def test_prompt_is_utf8_stdin_not_command_and_success_does_not_kill_completed_process(self):
        child = self.child()
        prompt = '用户问题：解释 CT 胜率；不要执行 & del "private"'
        with patch("src.csdemo.roundcast_chat.subprocess.Popen", return_value=child) as launch:
            self.assertEqual(self.runner.run(prompt, self.cancel), "当前尚未运行预测。")
        args, kwargs = launch.call_args
        self.assertIsInstance(args[0], list)
        self.assertNotIn(prompt, args[0])
        self.assertEqual(args[0][-1], "-")
        self.assertFalse(kwargs.get("shell", False))
        self.assertEqual(kwargs["stderr"], subprocess.DEVNULL)
        self.assertIn("roundcast-chat-", kwargs["cwd"])
        self.assertEqual(child.communicate.call_args_list[0].kwargs["input"], prompt.encode("utf-8"))
        child.kill.assert_not_called()

    def test_timeout_kills_only_the_owned_child_and_returns_no_partial_answer(self):
        self.runner.timeout = 0
        child = self.child()
        child.poll.return_value = None
        with patch("src.csdemo.roundcast_chat.subprocess.Popen", return_value=child) as launch:
            with self.assertRaises(ChatError) as caught:
                self.runner.run("解释", self.cancel)
        self.assertEqual(caught.exception.code, "timeout")
        self.assertNotIn("private", str(caught.exception))
        launch.assert_called_once()
        child.kill.assert_called_once_with()
        child.communicate.assert_called_once_with()

    def test_cancel_during_wait_kills_owned_child_and_never_returns_late_success(self):
        child = self.child()
        child.poll.return_value = None

        def interrupted_communication(*args, **kwargs):
            self.cancel.set()
            raise subprocess.TimeoutExpired("codex-test.exe", .25)

        # A callable side effect controls the cancellation at the actual wait boundary.
        calls = 0

        def communicate(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return interrupted_communication(*args, **kwargs)
            return completed_output("late reply"), None

        child.communicate.side_effect = communicate
        with patch("src.csdemo.roundcast_chat.subprocess.Popen", return_value=child):
            with self.assertRaises(ChatError) as caught:
                self.runner.run("解释", self.cancel)
        self.assertEqual(caught.exception.code, "cancelled")
        child.kill.assert_called_once_with()
        self.assertEqual(calls, 2)

    def test_pre_cancelled_job_does_not_launch_any_process(self):
        self.cancel.set()
        with patch("src.csdemo.roundcast_chat.subprocess.Popen") as launch:
            with self.assertRaises(ChatError) as caught:
                self.runner.run("解释", self.cancel)
        self.assertEqual(caught.exception.code, "cancelled")
        launch.assert_not_called()

    def test_retry_after_communicate_timeout_does_not_resend_stdin(self):
        child = self.child()
        child.communicate.side_effect = [subprocess.TimeoutExpired("codex-test.exe", .25),
                                         (completed_output("回复完成"), None), (b"", None)]
        with patch("src.csdemo.roundcast_chat.subprocess.Popen", return_value=child):
            self.assertEqual(self.runner.run("原始问题", self.cancel), "回复完成")
        calls = child.communicate.call_args_list
        self.assertEqual(calls[0].kwargs["input"], "原始问题".encode("utf-8"))
        self.assertIsNone(calls[1].kwargs["input"])
        child.kill.assert_not_called()

    def test_invalid_nonzero_failed_and_incomplete_streams_never_become_answers(self):
        failed = completed_output("untrusted partial answer") + b'\n{"type":"turn.failed"}'
        message_only = b'{"type":"item.completed","item":{"type":"agent_message","text":"partial"}}'
        for output, returncode in ((b"invalid json C:/private/auth.json", 0),
                                   (completed_output(), 1), (failed, 0), (message_only, 0),
                                   (b'{"type":"turn.completed"}', 0),
                                   (completed_output("   "), 0), (completed_output("a" * 16001), 0)):
            with self.subTest(returncode=returncode, output_length=len(output)):
                child = self.child(output, returncode)
                with patch("src.csdemo.roundcast_chat.subprocess.Popen", return_value=child):
                    with self.assertRaises(ChatError) as caught:
                        self.runner.run("解释", self.cancel)
                self.assertNotIn("private", str(caught.exception))
                self.assertNotIn("partial", str(caught.exception))
                child.kill.assert_not_called()

    def test_only_final_agent_text_is_returned_not_reasoning_or_tool_output(self):
        events = [
            {"type": "item.completed", "item": {"type": "reasoning", "text": "private reasoning"}},
            {"type": "item.completed", "item": {"type": "command_execution", "text": "private output"}},
            {"type": "item.completed", "item": {"type": "agent_message", "text": "正在分析"}},
            {"type": "item.completed", "item": {"type": "agent_message", "text": " 最终解释 "}},
            {"type": "turn.completed"},
        ]
        output = "\n".join(json.dumps(event, ensure_ascii=False) for event in events).encode("utf-8")
        child = self.child(output)
        with patch("src.csdemo.roundcast_chat.subprocess.Popen", return_value=child):
            self.assertEqual(self.runner.run("解释", self.cancel), "最终解释")

    def test_process_start_failure_is_sanitized(self):
        with patch("src.csdemo.roundcast_chat.subprocess.Popen", side_effect=OSError("C:/private/secret")):
            with self.assertRaises(ChatError) as caught:
                self.runner.run("解释", self.cancel)
        self.assertEqual(caught.exception.code, "start_failed")
        self.assertNotIn("private", str(caught.exception))
        self.assertNotIn("secret", str(caught.exception))

    def test_missing_installation_and_failed_login_status_do_not_expose_diagnostics(self):
        self.runner.executable = None
        with patch("src.csdemo.roundcast_chat.subprocess.run") as run:
            self.assertFalse(self.runner.status()["available"])
        run.assert_not_called()
        self.runner.executable = "codex-test.exe"
        failures = (subprocess.CompletedProcess([], 1, stdout=b"private-token", stderr=b"private auth.json"),
                    OSError("private auth.json"), subprocess.TimeoutExpired("private", 10))
        for failure in failures:
            with self.subTest(failure=type(failure).__name__):
                kwargs = {"side_effect": failure} if isinstance(failure, Exception) else {"return_value": failure}
                with patch("src.csdemo.roundcast_chat.subprocess.run", **kwargs):
                    status = self.runner.status()
                self.assertFalse(status["available"])
                self.assertNotIn("private", json.dumps(status))
                self.assertNotIn("auth.json", json.dumps(status))


if __name__ == "__main__":
    unittest.main()
