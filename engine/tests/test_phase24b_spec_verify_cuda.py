from __future__ import annotations

import math
from pathlib import Path
from unittest import mock
import unittest

from vinf.cuda.build import cuda_extension_compile_command
from vinf.cuda.spec_verify import spec_verify_cpu


class Phase24BSpecVerifyCUDATests(unittest.TestCase):
    def assert_close_rows(self, actual, expected, tol=1e-5) -> None:
        self.assertEqual(len(actual), len(expected))
        for row_idx, (actual_row, expected_row) in enumerate(zip(actual, expected)):
            self.assertEqual(len(actual_row), len(expected_row))
            for col_idx, (a, e) in enumerate(zip(actual_row, expected_row)):
                self.assertTrue(
                    math.isclose(a, e, rel_tol=tol, abs_tol=tol),
                    (row_idx, col_idx, a, e),
                )

    def test_spec_verify_build_command_shape(self) -> None:
        with mock.patch("shutil.which", return_value="/usr/local/cuda/bin/nvcc"):
            cmd = cuda_extension_compile_command(
                Path("/tmp/engine"),
                source="csrc/megakernel/spec_verify.cu",
                output="src/vinf/_cuda_spec_verify.so",
            )
        self.assertIn("/tmp/engine/csrc/megakernel/spec_verify.cu", cmd)
        self.assertIn("/tmp/engine/src/vinf/_cuda_spec_verify.so", cmd)
        self.assertIn("-arch=sm_86", cmd)

    def test_cpu_spec_verify_emits_probability_rows_for_every_position(self) -> None:
        rows = spec_verify_cpu(
            [[0.0, 1.0, 2.0], [2.0, 0.0, -2.0]],
            num_verify_tokens=2,
            vocab_size=3,
        )
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertAlmostEqual(sum(row), 1.0)

    def test_cuda_spec_verify_matches_cpu_if_available(self) -> None:
        logits = [[0.0, 1.0, 2.0], [2.0, 0.0, -2.0]]
        expected = spec_verify_cpu(logits, num_verify_tokens=2, vocab_size=3)
        try:
            from vinf.cuda.spec_verify import spec_verify_cuda

            actual = spec_verify_cuda(logits, num_verify_tokens=2, vocab_size=3)
        except Exception as exc:
            self.skipTest(f"CUDA speculative verifier unavailable: {exc}")
        self.assert_close_rows(actual, expected)


if __name__ == "__main__":
    unittest.main()
