from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil

from vinf.errors import ExecutorUnavailableError


@dataclass(frozen=True, slots=True)
class CUDABuildConfig:
    target_gpu: str = "RTX3080TI"
    compute_capability: str = "sm_86"
    cxx_standard: str = "c++20"
    use_fast_math: bool = True
    source: Path = Path("csrc/megakernel/noop_smoke.cu")
    output: Path = Path("src/vinf/_cuda_noop.so")

    @property
    def defines(self) -> tuple[str, ...]:
        return (
            "-DVINF_TARGET_RTX_3080_TI=1",
            "-DVINF_AMPERE=1",
            "-DVINF_DISABLE_TMA=1",
            "-DVINF_DISABLE_HOPPER=1",
            "-DVINF_DISABLE_BLACKWELL=1",
        )

    @property
    def arch_flag(self) -> str:
        return f"-arch={self.compute_capability}"


def cuda_smoke_compile_command(
    project_root: str | Path, config: CUDABuildConfig | None = None
) -> list[str]:
    config = config or CUDABuildConfig()
    root = Path(project_root)
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise ExecutorUnavailableError("nvcc not found on PATH; cannot build CUDA smoke target")
    cmd = [
        nvcc,
        str(root / config.source),
        "-shared",
        "-Xcompiler=-fPIC",
        config.arch_flag,
        f"-std={config.cxx_standard}",
        *python_includes(),
        "-I",
        str(root / "csrc/common"),
        "-cudart",
        "static",  # bake the CUDA runtime in: release builds need only the NVIDIA driver
        "-o",
        str(root / config.output),
    ]
    if config.use_fast_math:
        cmd.append("--use_fast_math")
    cmd.extend(config.defines)
    return cmd


def cuda_extension_compile_command(
    project_root: str | Path,
    *,
    source: str,
    output: str,
    config: CUDABuildConfig | None = None,
    extra_flags: tuple[str, ...] = (),
) -> list[str]:
    config = config or CUDABuildConfig()
    root = Path(project_root)
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise ExecutorUnavailableError("nvcc not found on PATH; cannot build CUDA target")
    cmd = [
        nvcc,
        str(root / source),
        "-shared",
        "-Xcompiler=-fPIC",
        config.arch_flag,
        f"-std={config.cxx_standard}",
        *python_includes(),
        "-I",
        str(root / "csrc/common"),
        "-cudart",
        "static",  # bake the CUDA runtime in: release builds need only the NVIDIA driver
        "-o",
        str(root / output),
    ]
    if config.use_fast_math:
        cmd.append("--use_fast_math")
    cmd.extend(config.defines)
    cmd.extend(extra_flags)
    return cmd


def python_includes() -> list[str]:
    import sysconfig

    include = sysconfig.get_paths()["include"]
    platinclude = sysconfig.get_paths().get("platinclude", include)
    flags = ["-I", include]
    if platinclude != include:
        flags.extend(["-I", platinclude])
    return flags
