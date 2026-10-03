from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from vinf.cuda.build import cuda_extension_compile_command
from vinf.cuda.gguf_upload import upload_checksum_cpu


class Phase27GGUFUploadTests(unittest.TestCase):
    def test_cuda_upload_compile_command_shape(self) -> None:
        with mock.patch("shutil.which", return_value="/usr/local/cuda/bin/nvcc"):
            cmd = cuda_extension_compile_command(
                Path("/tmp/engine"),
                source="csrc/megakernel/gguf_upload.cu",
                output="src/vinf/_cuda_gguf_upload.so",
            )
        self.assertIn("/tmp/engine/csrc/megakernel/gguf_upload.cu", cmd)
        self.assertIn("/tmp/engine/src/vinf/_cuda_gguf_upload.so", cmd)
        self.assertIn("-arch=sm_86", cmd)

    def test_upload_checksum_cpu_matches_byte_sum(self) -> None:
        self.assertEqual(upload_checksum_cpu(bytes(range(16))), (16, 120))

    def test_cuda_upload_checksum_if_available(self) -> None:
        from vinf.cuda.gguf_upload import upload_checksum_cuda

        data = bytes(range(16))
        try:
            self.assertEqual(upload_checksum_cuda(data), upload_checksum_cpu(data))
        except (ImportError, RuntimeError) as exc:
            self.skipTest(f"CUDA GGUF upload unavailable: {exc}")


if __name__ == "__main__":
    unittest.main()
