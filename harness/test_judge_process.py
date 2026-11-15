"""Offline checks: no live Pi child, credential read, or benchmark cancellation."""
import os
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import judge


class HeadlessJudgeTests(unittest.TestCase):
    def test_judge_uses_piped_consoleless_worker(self):
        with tempfile.TemporaryDirectory() as scratch:
            with patch.object(judge.subprocess, "run") as run:
                run.return_value.stdout = '{"steps": []}'
                result = judge.call("xai/grok-4.7", "xhigh", "synthetic fixture", scratch)
            self.assertEqual(result, '{"steps": []}')
            args, kwargs = run.call_args
            self.assertIn("--no-session", args[0])
            self.assertTrue(kwargs["capture_output"])
            self.assertEqual(kwargs["creationflags"], subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            self.assertEqual(kwargs["env"]["PI_OFFLINE"], "1")

    def test_numbered_bridge_worker_does_not_inherit_interactive_tools(self):
        with tempfile.TemporaryDirectory() as scratch:
            with patch.object(judge.subprocess, "run") as run:
                run.return_value.stdout = "fixture"
                judge.call("claude-bridge-2/claude-sonnet-5-5", "high", "synthetic fixture", scratch)
            args = run.call_args.args[0]
            self.assertIn("--no-tools", args)
            self.assertIn("--no-session", args)
            self.assertIn("-ne", args)
            self.assertIn("-e", args)


if __name__ == "__main__":
    unittest.main()
