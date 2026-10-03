from __future__ import annotations

from dataclasses import dataclass
import subprocess

from vinf.errors import UnsupportedHardwareError


@dataclass(frozen=True, slots=True)
class CUDADeviceInfo:
    name: str
    compute_capability: str
    memory_total_mib: int


def query_nvidia_smi() -> list[CUDADeviceInfo]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,compute_cap,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            cmd,
            check=True,
            text=True,
            capture_output=True,
        )
    except FileNotFoundError as exc:
        raise UnsupportedHardwareError("nvidia-smi not found") from exc
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() or exc.stdout.strip() or "nvidia-smi failed"
        raise UnsupportedHardwareError(message) from exc

    devices = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            raise UnsupportedHardwareError(f"unexpected nvidia-smi output: {line}")
        devices.append(
            CUDADeviceInfo(
                name=parts[0],
                compute_capability=parts[1],
                memory_total_mib=int(parts[2]),
            )
        )
    return devices


def check_rtx_3080_ti() -> CUDADeviceInfo:
    devices = query_nvidia_smi()
    for device in devices:
        if "3080 Ti" in device.name and device.compute_capability == "8.6":
            return device
    seen = ", ".join(f"{d.name} cc={d.compute_capability}" for d in devices) or "none"
    raise UnsupportedHardwareError(f"RTX 3080 Ti compute capability 8.6 not found; saw {seen}")

