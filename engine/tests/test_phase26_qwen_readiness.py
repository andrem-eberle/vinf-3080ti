from __future__ import annotations

import os

import subprocess
import sys
import unittest
from pathlib import Path

from tests.test_phase5_gguf import write_tiny_gguf
from vinf.gguf.parser import load_gguf
from vinf.gguf.qwen import probe_qwen_gguf, require_qwen_inference_ready


QWEN_GGUF = Path(os.environ.get("VINF_QWEN_GGUF", "/home/z/mt2/Qwen3.8-27B-UD-Q3_K_XL.gguf"))


class Phase26QwenReadinessTests(unittest.TestCase):
    def test_probe_command_runs_on_tiny_gguf(self) -> None:
        completed = subprocess.run(
            [sys.executable, "scripts/probe_gguf.py", str(write_tiny_gguf())],
            cwd=Path(__file__).resolve().parents[1],
            env={"PYTHONPATH": "src"},
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertIn("architecture=llama", completed.stdout)
        self.assertIn("supported_for_inference=false", completed.stdout)

    def test_real_qwen_metadata_probe_if_file_exists(self) -> None:
        if not QWEN_GGUF.exists():
            self.skipTest("local Qwen GGUF is not present")
        report = probe_qwen_gguf(load_gguf(QWEN_GGUF))
        self.assertEqual(report.architecture, "qwen35")
        self.assertEqual(report.metadata.num_hidden_layers, 65)
        self.assertEqual(report.metadata.hidden_size, 5120)
        self.assertEqual(report.metadata.num_attention_heads, 24)
        self.assertEqual(report.metadata.num_kv_heads, 4)
        self.assertEqual(report.metadata.vocab_size, 248320)
        self.assertEqual(report.metadata.dtype, "quantized")
        self.assertGreater(report.quantized_tensor_count, 0)
        self.assertIn("Q4_K", report.tensor_type_counts)
        self.assertIn("Q6_K", report.tensor_type_counts)
        self.assertTrue(report.supported_for_inference)
        self.assertTrue(report.feature_fields_present)

    def test_real_qwen_is_ready_for_loading_if_file_exists(self) -> None:
        if not QWEN_GGUF.exists():
            self.skipTest("local Qwen GGUF is not present")
        report = require_qwen_inference_ready(load_gguf(QWEN_GGUF))
        self.assertTrue(report.supported_for_inference)


if __name__ == "__main__":
    unittest.main()
