from __future__ import annotations

import unittest

from vinf.gguf.parser import GGUFTensorInfo, GGUFTensorType
from vinf.gguf.qwen_tensors import map_qwen_tensor_name
from vinf.gguf.residency import (
    LayerWeightCache,
    PinnedCPUStagingBuffer,
    TensorResidency,
    plan_qwen_tensor_residency,
    stream_gpu_uploads,
)


def tensor(name: str, dims: tuple[int, ...], typ: GGUFTensorType) -> GGUFTensorInfo:
    return GGUFTensorInfo(
        name=name,
        dimensions=dims,
        tensor_type=typ,
        relative_offset=0,
        absolute_offset=0,
    )


class _Closable:
    def close(self) -> None:
        pass


class _FakeGGUF:
    def __init__(self) -> None:
        self.tensors = {
            "output_norm.weight": tensor("output_norm.weight", (8,), GGUFTensorType.F16),
            "blk.0.attn_norm.weight": tensor("blk.0.attn_norm.weight", (8,), GGUFTensorType.F16),
            "blk.0.ffn_up.weight": tensor("blk.0.ffn_up.weight", (8, 16), GGUFTensorType.Q4_K),
        }
        self.read_names: list[str] = []

    def mmap_tensor(self, name: str):
        self.read_names.append(name)
        return _Closable(), _Closable(), memoryview(bytes(self.tensors[name].nbytes))


class Phase27ResidencyTests(unittest.TestCase):
    def test_mixed_residency_keeps_large_layer_weights_streamed(self) -> None:
        gguf = _FakeGGUF()
        plan = plan_qwen_tensor_residency(
            gguf,
            map_name=map_qwen_tensor_name,
            gpu_budget_bytes=1024,
            reserved_gpu_bytes=0,
        )
        self.assertTrue(plan.fits_gpu_budget)
        self.assertEqual(plan.entry_for("output_norm.weight").residency, TensorResidency.GPU)
        self.assertEqual(plan.entry_for("blk.0.attn_norm.weight").residency, TensorResidency.GPU)
        self.assertEqual(plan.entry_for("blk.0.ffn_up.weight").residency, TensorResidency.CPU_MMAP)

    def test_stream_uploads_read_only_gpu_resident_tensor_ranges(self) -> None:
        gguf = _FakeGGUF()
        plan = plan_qwen_tensor_residency(
            gguf,
            map_name=map_qwen_tensor_name,
            gpu_budget_bytes=1024,
            reserved_gpu_bytes=0,
        )
        uploaded: list[tuple[str, int]] = []
        records = stream_gpu_uploads(
            gguf,
            plan,
            PinnedCPUStagingBuffer(64),
            lambda name, data: uploaded.append((name, len(data))) or sum(data.cast("B")),
        )
        self.assertEqual(gguf.read_names, ["blk.0.attn_norm.weight", "output_norm.weight"])
        self.assertEqual([record.name for record in records], gguf.read_names)
        self.assertEqual([record.checksum for record in records], [0, 0])
        self.assertEqual(uploaded, [("layers.0.attn_norm.weight", 16), ("norm.weight", 16)])

    def test_staging_buffer_rejects_tensors_that_do_not_fit(self) -> None:
        with self.assertRaises(ValueError):
            PinnedCPUStagingBuffer(4).stage(memoryview(bytes(8)))

    def test_layer_weight_cache_eviction_is_lru(self) -> None:
        cache = LayerWeightCache(10)
        self.assertEqual(cache.touch("layer0", 4), ())
        self.assertEqual(cache.touch("layer1", 4), ())
        self.assertEqual(cache.touch("layer2", 4), ("layer0",))
        self.assertEqual(cache.names, ("layer1", "layer2"))
        self.assertEqual(cache.used_bytes, 8)


if __name__ == "__main__":
    unittest.main()
