from __future__ import annotations


def upload_checksum_cpu(data: bytes | bytearray | memoryview) -> tuple[int, int]:
    view = memoryview(data).cast("B")
    return len(view), sum(int(byte) for byte in view)


def upload_checksum_cuda(data: bytes | bytearray | memoryview) -> tuple[int, int]:
    from vinf import _cuda_gguf_upload

    nbytes, checksum = _cuda_gguf_upload.upload_checksum(bytes(memoryview(data)))
    return int(nbytes), int(checksum)
