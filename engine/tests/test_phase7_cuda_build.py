from __future__ import annotations

from pathlib import Path
from unittest import mock
import subprocess
import unittest

from vinf.cuda.build import CUDABuildConfig, cuda_smoke_compile_command
from vinf.cuda.hardware import CUDADeviceInfo, check_rtx_3080_ti, query_nvidia_smi
from vinf.errors import ExecutorUnavailableError, UnsupportedHardwareError


class Phase7CUDABuildTests(unittest.TestCase):
    def test_build_config_targets_sm86_and_gates_newer_features(self) -> None:
        config = CUDABuildConfig()
        self.assertEqual(config.compute_capability, "sm_86")
        self.assertEqual(config.arch_flag, "-arch=sm_86")
        self.assertIn("-DVINF_TARGET_RTX_3080_TI=1", config.defines)
        self.assertIn("-DVINF_DISABLE_TMA=1", config.defines)
        self.assertIn("-DVINF_DISABLE_HOPPER=1", config.defines)
        self.assertIn("-DVINF_DISABLE_BLACKWELL=1", config.defines)

    def test_cuda_smoke_compile_command_requires_nvcc(self) -> None:
        with mock.patch("shutil.which", return_value=None):
            with self.assertRaises(ExecutorUnavailableError):
                cuda_smoke_compile_command(Path("."))

    def test_cuda_smoke_compile_command_shape(self) -> None:
        with mock.patch("shutil.which", return_value="/usr/local/cuda/bin/nvcc"):
            cmd = cuda_smoke_compile_command(Path("/tmp/engine"))
        self.assertIn("-arch=sm_86", cmd)
        self.assertIn("-std=c++20", cmd)
        self.assertIn("-DVINF_AMPERE=1", cmd)
        self.assertIn("/tmp/engine/csrc/megakernel/noop_smoke.cu", cmd)

    def test_config_header_contains_rtx_3080_ti_values(self) -> None:
        header = Path("csrc/common/config_3080ti.cuh").read_text()
        self.assertIn("VINF_CUDA_ARCH_SM86 86", header)
        self.assertIn("VINF_RTX_3080_TI_EXPECTED_SMS 80", header)
        self.assertIn("VINF_DISABLE_TMA 1", header)

    def test_query_nvidia_smi_parses_devices(self) -> None:
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="NVIDIA GeForce RTX 3080 Ti, 8.6, 12288\n",
            stderr="",
        )
        with mock.patch("subprocess.run", return_value=completed):
            devices = query_nvidia_smi()
        self.assertEqual(
            devices,
            [CUDADeviceInfo("NVIDIA GeForce RTX 3080 Ti", "8.6", 12288)],
        )

    def test_check_rtx_3080_ti_accepts_expected_device(self) -> None:
        with mock.patch(
            "vinf.cuda.hardware.query_nvidia_smi",
            return_value=[CUDADeviceInfo("NVIDIA GeForce RTX 3080 Ti", "8.6", 12288)],
        ):
            self.assertEqual(check_rtx_3080_ti().compute_capability, "8.6")

    def test_check_rtx_3080_ti_fails_clearly_for_wrong_device(self) -> None:
        with mock.patch(
            "vinf.cuda.hardware.query_nvidia_smi",
            return_value=[CUDADeviceInfo("NVIDIA GeForce RTX 4090", "8.9", 24576)],
        ):
            with self.assertRaises(UnsupportedHardwareError):
                check_rtx_3080_ti()


if __name__ == "__main__":
    unittest.main()

