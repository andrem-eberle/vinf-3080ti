from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from enum import StrEnum
from typing import Callable

from vinf.errors import InsufficientMemoryError
from vinf.gguf.parser import GGUFFile, GGUFTensorInfo


class TensorResidency(StrEnum):
    CPU_MMAP = "cpu_mmap"
    GPU = "gpu"
    PINNED_STREAM = "pinned_stream"


@dataclass(frozen=True, slots=True)
class TensorResidencyEntry:
    name: str
    internal_name: str
    nbytes: int
    residency: TensorResidency
    reason: str


@dataclass(frozen=True, slots=True)
class TensorResidencyPlan:
    entries: tuple[TensorResidencyEntry, ...]
    gpu_budget_bytes: int
    reserved_gpu_bytes: int

    @property
    def gpu_bytes(self) -> int:
        return sum(entry.nbytes for entry in self.entries if entry.residency is TensorResidency.GPU)

    @property
    def cpu_mmap_bytes(self) -> int:
        return sum(entry.nbytes for entry in self.entries if entry.residency is TensorResidency.CPU_MMAP)

    @property
    def fits_gpu_budget(self) -> bool:
        return self.gpu_bytes <= self.gpu_budget_bytes - self.reserved_gpu_bytes

    def entry_for(self, name: str) -> TensorResidencyEntry:
        for entry in self.entries:
            if entry.name == name:
                return entry
        raise KeyError(name)


@dataclass(frozen=True, slots=True)
class UploadRecord:
    name: str
    internal_name: str
    nbytes: int
    checksum: int | None = None


class PinnedCPUStagingBuffer:
    def __init__(self, capacity_bytes: int) -> None:
        if capacity_bytes <= 0:
            raise ValueError("staging buffer capacity must be positive")
        self.capacity_bytes = capacity_bytes
        self._buffer = bytearray(capacity_bytes)

    def stage(self, data: memoryview) -> memoryview:
        if len(data) > self.capacity_bytes:
            raise ValueError("tensor is larger than the staging buffer")
        self._buffer[: len(data)] = data
        return memoryview(self._buffer)[: len(data)]


class LayerWeightCache:
    def __init__(self, capacity_bytes: int) -> None:
        if capacity_bytes <= 0:
            raise ValueError("cache capacity must be positive")
        self.capacity_bytes = capacity_bytes
        self._entries: OrderedDict[str, int] = OrderedDict()
        self._used_bytes = 0

    @property
    def used_bytes(self) -> int:
        return self._used_bytes

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._entries)

    def touch(self, name: str, nbytes: int) -> tuple[str, ...]:
        if nbytes > self.capacity_bytes:
            raise ValueError("layer is larger than the cache capacity")
        evicted: list[str] = []
        if name in self._entries:
            self._used_bytes -= self._entries.pop(name)
        self._entries[name] = nbytes
        self._used_bytes += nbytes
        self._entries.move_to_end(name)
        while self._used_bytes > self.capacity_bytes:
            evicted_name, evicted_bytes = self._entries.popitem(last=False)
            self._used_bytes -= evicted_bytes
            evicted.append(evicted_name)
        return tuple(evicted)


def plan_qwen_tensor_residency(
    gguf: GGUFFile,
    *,
    map_name: Callable[[str], str],
    gpu_budget_bytes: int = 12 * 1024**3,
    reserved_gpu_bytes: int = 2 * 1024**3,
) -> TensorResidencyPlan:
    usable_gpu_bytes = gpu_budget_bytes - reserved_gpu_bytes
    used_gpu_bytes = 0
    entries: list[TensorResidencyEntry] = []
    for tensor in sorted(gguf.tensors.values(), key=_tensor_priority):
        residency = TensorResidency.CPU_MMAP
        reason = "streamed from CPU mmap"
        if _prefer_gpu_residency(tensor) and used_gpu_bytes + tensor.nbytes <= usable_gpu_bytes:
            residency = TensorResidency.GPU
            reason = "small or frequently reused tensor"
            used_gpu_bytes += tensor.nbytes
        entries.append(
            TensorResidencyEntry(
                name=tensor.name,
                internal_name=map_name(tensor.name),
                nbytes=tensor.nbytes,
                residency=residency,
                reason=reason,
            )
        )
    return TensorResidencyPlan(
        entries=tuple(sorted(entries, key=lambda entry: entry.name)),
        gpu_budget_bytes=gpu_budget_bytes,
        reserved_gpu_bytes=reserved_gpu_bytes,
    )


def stream_gpu_uploads(
    gguf: GGUFFile,
    plan: TensorResidencyPlan,
    staging: PinnedCPUStagingBuffer,
    upload: Callable[[str, memoryview], int | None],
) -> tuple[UploadRecord, ...]:
    records: list[UploadRecord] = []
    for entry in plan.entries:
        if entry.residency is not TensorResidency.GPU:
            continue
        file_obj, mm, view = gguf.mmap_tensor(entry.name)
        try:
            staged = staging.stage(view)
            checksum = upload(entry.internal_name, staged)
            records.append(
                UploadRecord(
                    name=entry.name,
                    internal_name=entry.internal_name,
                    nbytes=len(staged),
                    checksum=checksum,
                )
            )
        finally:
            view.release()
            mm.close()
            file_obj.close()
    return tuple(records)


def _tensor_priority(tensor: GGUFTensorInfo) -> tuple[int, str]:
    if tensor.name in {"output_norm.weight", "token_embd.weight", "output.weight"}:
        return (0, tensor.name)
    if tensor.name.endswith("_norm.weight") or tensor.name.endswith("norm.weight"):
        return (1, tensor.name)
    return (2, tensor.name)


def _prefer_gpu_residency(tensor: GGUFTensorInfo) -> bool:
    return (
        tensor.name in {"output_norm.weight", "token_embd.weight", "output.weight"}
        or tensor.name.endswith("_norm.weight")
        or tensor.name.endswith("norm.weight")
    )


@dataclass(frozen=True, slots=True)
class QwenDecodeReservation:
    kv_cache_bytes: int
    ssm_state_bytes: int
    activation_bytes: int
    staging_bytes: int
    safety_bytes: int

    @property
    def total(self) -> int:
        return (
            self.kv_cache_bytes
            + self.ssm_state_bytes
            + self.activation_bytes
            + self.staging_bytes
            + self.safety_bytes
        )


@dataclass(frozen=True, slots=True)
class QwenDecodeResidencyPlan:
    """Placement of decode weights: GPU-resident, pinned-host streamed, or CPU mmap."""

    entries: tuple[TensorResidencyEntry, ...]
    free_vram_bytes: int
    reservation: QwenDecodeReservation
    max_context: int

    @property
    def weight_budget_bytes(self) -> int:
        return self.free_vram_bytes - self.reservation.total

    def bytes_for(self, residency: TensorResidency) -> int:
        return sum(entry.nbytes for entry in self.entries if entry.residency is residency)

    @property
    def gpu_bytes(self) -> int:
        return self.bytes_for(TensorResidency.GPU)

    @property
    def streamed_bytes(self) -> int:
        return self.bytes_for(TensorResidency.PINNED_STREAM)

    def names(self, residency: TensorResidency) -> tuple[str, ...]:
        return tuple(entry.name for entry in self.entries if entry.residency is residency)

    def report(self) -> str:
        gib = 1024**3
        r = self.reservation
        lines = [
            f"free VRAM            {self.free_vram_bytes / gib:7.2f} GiB",
            f"  KV cache           {r.kv_cache_bytes / gib:7.2f} GiB (max_context={self.max_context})",
            f"  SSM state          {r.ssm_state_bytes / gib:7.2f} GiB",
            f"  activations        {r.activation_bytes / gib:7.2f} GiB",
            f"  stream staging     {r.staging_bytes / gib:7.2f} GiB",
            f"  safety margin      {r.safety_bytes / gib:7.2f} GiB",
            f"weight budget        {self.weight_budget_bytes / gib:7.2f} GiB",
            f"GPU-resident weights {self.gpu_bytes / gib:7.2f} GiB "
            f"({len(self.names(TensorResidency.GPU))} tensors)",
            f"streamed per token   {self.streamed_bytes / gib:7.2f} GiB "
            f"({len(self.names(TensorResidency.PINNED_STREAM))} tensors, pinned host)",
            f"CPU mmap             {self.bytes_for(TensorResidency.CPU_MMAP) / gib:7.2f} GiB "
            f"({', '.join(self.names(TensorResidency.CPU_MMAP))})",
        ]
        return "\n".join(lines)


def is_elementwise_weight(tensor: GGUFTensorInfo) -> bool:
    """F32 tensors consumed by elementwise kernels (norms, ssm_a, dt bias, conv1d) rather than matvec."""
    return len(tensor.dimensions) == 1 or tensor.name.endswith("ssm_conv1d.weight")


def plan_qwen_decode_residency(
    gguf: GGUFFile,
    layer_names: tuple[str, ...],
    *,
    free_vram_bytes: int,
    kv_cache_bytes: int,
    ssm_state_bytes: int,
    activation_bytes: int,
    safety_bytes: int = 512 * 1024**2,
    max_context: int,
    stream_slots: int = 3,
    priority_names: tuple[str, ...] = (),
    map_name: Callable[[str], str] = lambda name: name,
) -> QwenDecodeResidencyPlan:
    """Fill live free VRAM with decode weights; stream the rest from pinned host memory.

    Every decode weight is read once per token, so any resident byte saves one streamed
    byte. Elementwise F32 weights must be resident; the output head goes next (largest
    single per-token tensor, and it sizes the staging buffer if streamed); then layer
    matrices in layer order. The token embedding stays in CPU mmap (row lookups only).
    Streaming reserves `stream_slots` staging buffers so copies run ahead of compute.
    `priority_names` (e.g. the MTP draft block) are placed right after the output head and must
    be GPU-resident; their use pattern does not follow the per-token stream order.
    """
    if stream_slots < 1:
        raise ValueError("stream_slots must be >= 1")
    if "token_embd.weight" not in gguf.tensors:
        raise KeyError("token_embd.weight")
    decode = [name for name in layer_names] + ["output_norm.weight", "output.weight"] + list(priority_names)
    for name in decode:
        if name not in gguf.tensors:
            raise KeyError(name)
    mandatory = [name for name in decode if is_elementwise_weight(gguf.tensors[name])]
    matrices = [name for name in decode if name not in mandatory]
    priority = set(priority_names)
    matrices.sort(key=lambda name: 0 if name == "output.weight" else 1 if name in priority else 2)  # stable

    largest_layer_matrix = stream_slots * max(
        (gguf.tensors[name].nbytes for name in matrices if name != "output.weight"), default=0
    )
    fixed = kv_cache_bytes + ssm_state_bytes + activation_bytes + safety_bytes
    mandatory_bytes = sum(gguf.tensors[name].nbytes for name in mandatory)
    budget = free_vram_bytes - fixed - largest_layer_matrix - mandatory_bytes
    if budget < 0:
        raise InsufficientMemoryError(
            f"insufficient VRAM: {free_vram_bytes / 1024**3:.2f} GiB free, need at least "
            f"{(fixed + largest_layer_matrix + mandatory_bytes) / 1024**3:.2f} GiB for caches, "
            f"activations, staging, and elementwise weights; lower max_context"
        )

    placement: dict[str, tuple[TensorResidency, str]] = {
        name: (TensorResidency.GPU, "elementwise weight") for name in mandatory
    }
    staging = largest_layer_matrix
    used = 0
    for name in matrices:
        nbytes = gguf.tensors[name].nbytes
        extra_staging = max(0, stream_slots * nbytes - staging) if name == "output.weight" else 0
        if used + nbytes <= budget:
            placement[name] = (TensorResidency.GPU, "fits VRAM budget")
            used += nbytes
        else:
            placement[name] = (TensorResidency.PINNED_STREAM, "streamed from pinned host memory")
            if extra_staging:
                if budget - used < extra_staging:
                    raise InsufficientMemoryError("insufficient VRAM to stage the output head")
                staging += extra_staging
                budget -= extra_staging
    unplaced = [name for name in priority_names if placement[name][0] is not TensorResidency.GPU]
    if unplaced:
        raise InsufficientMemoryError(
            "insufficient VRAM to keep priority tensors resident: " + ", ".join(sorted(unplaced))
        )
    if not any(res is TensorResidency.PINNED_STREAM for res, _ in placement.values()):
        staging = 0
    placement["token_embd.weight"] = (TensorResidency.CPU_MMAP, "token rows dequantized on CPU")

    entries = tuple(
        TensorResidencyEntry(
            name=name,
            internal_name=map_name(name),
            nbytes=gguf.tensors[name].nbytes,
            residency=residency,
            reason=reason,
        )
        for name, (residency, reason) in sorted(placement.items())
    )
    return QwenDecodeResidencyPlan(
        entries=entries,
        free_vram_bytes=free_vram_bytes,
        reservation=QwenDecodeReservation(
            kv_cache_bytes=kv_cache_bytes,
            ssm_state_bytes=ssm_state_bytes,
            activation_bytes=activation_bytes,
            staging_bytes=staging,
            safety_bytes=safety_bytes,
        ),
        max_context=max_context,
    )
