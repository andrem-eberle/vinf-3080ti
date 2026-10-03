from __future__ import annotations

import unittest

from vinf.config import EngineConfig, SpeculativeConfig
from vinf.errors import ConfigurationError, DecodeError
from vinf.memory import BufferRegistry, BufferRole, BufferSpec, MemoryPlanner, Residency
from vinf.models.metadata import ModelArchitecture, ModelMetadata
from vinf.runtime.kv_cache import KVCacheSet, LogicalKVCache


def tiny_model(dtype: str = "fp16") -> ModelMetadata:
    return ModelMetadata(
        architecture=ModelArchitecture.LLAMA,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_kv_heads=2,
        hidden_size=256,
        intermediate_size=512,
        head_dim=64,
        vocab_size=1024,
        max_position_embeddings=256,
        dtype=dtype,
    )


class Phase2MemoryRuntimeTests(unittest.TestCase):
    def test_buffer_registry_tracks_gpu_bytes(self) -> None:
        registry = BufferRegistry()
        registry.add(
            BufferSpec(
                name="a",
                role=BufferRole.ACTIVATION,
                shape=(2, 4),
                dtype="fp16",
                residency=Residency.GPU,
            )
        )
        registry.add(
            BufferSpec(
                name="b",
                role=BufferRole.LOGITS,
                shape=(3,),
                dtype="fp32",
                residency=Residency.CPU,
            )
        )
        self.assertEqual(registry.gpu_bytes, 16)
        self.assertEqual(registry.cpu_bytes, 12)
        self.assertEqual(registry.by_role(BufferRole.ACTIVATION)[0].name, "a")
        with self.assertRaises(ConfigurationError):
            registry.add(
                BufferSpec(
                    name="a",
                    role=BufferRole.ACTIVATION,
                    shape=(1,),
                    dtype="fp16",
                )
            )

    def test_memory_planner_includes_core_gpu_buffers(self) -> None:
        config = EngineConfig(
            max_seq_len=128,
            speculative=SpeculativeConfig(enabled=True, gamma=3),
        )
        plan = MemoryPlanner(config).plan(tiny_model(), draft=tiny_model(), max_gamma=3)
        names = set(plan.registry.specs)
        self.assertIn("target_weights", names)
        self.assertIn("target_kv", names)
        self.assertIn("draft_weights", names)
        self.assertIn("draft_kv", names)
        self.assertIn("hidden_states", names)
        self.assertIn("logits", names)
        self.assertGreater(plan.planned_gpu_bytes, 0)
        self.assertTrue(plan.fits_gpu_budget)
        self.assertEqual(plan.registry.get("target_kv").shape[3], 131)

    def test_memory_planner_can_report_budget_failure(self) -> None:
        plan = MemoryPlanner(
            EngineConfig(max_seq_len=128),
            available_vram_bytes=1024,
            reserved_vram_bytes=0,
        ).plan(tiny_model())
        self.assertFalse(plan.fits_gpu_budget)
        self.assertLess(plan.remaining_gpu_bytes, 0)

    def test_target_kv_speculative_entries_are_invisible_until_commit(self) -> None:
        cache = LogicalKVCache("target", max_seq_len=8, max_speculative_tokens=3)
        cache.append_committed(4)
        cache.write_speculative(2)
        self.assertTrue(cache.can_read_position(3))
        self.assertFalse(cache.can_read_position(4))
        self.assertTrue(cache.can_read_position(4, include_speculative=True))
        cache.commit_speculative(1)
        self.assertEqual(cache.committed_len, 5)
        self.assertEqual(cache.spec_len, 0)
        self.assertTrue(cache.can_read_position(4))
        self.assertFalse(cache.can_read_position(5))

    def test_rejected_speculative_entries_are_not_readable_after_discard(self) -> None:
        cache = LogicalKVCache("target", max_seq_len=8, max_speculative_tokens=3)
        cache.append_committed(3)
        cache.write_speculative(3)
        self.assertTrue(cache.can_read_position(5, include_speculative=True))
        cache.discard_speculative()
        self.assertFalse(cache.can_read_position(3))
        self.assertFalse(cache.can_read_position(5, include_speculative=True))
        with self.assertRaises(DecodeError):
            cache.require_read_position(3)

    def test_draft_and_target_cache_set_discard_together(self) -> None:
        caches = KVCacheSet(
            target=LogicalKVCache("target", max_seq_len=8, max_speculative_tokens=2),
            draft=LogicalKVCache("draft", max_seq_len=8, max_speculative_tokens=2),
        )
        caches.target.append_committed(2)
        caches.draft.append_committed(2)
        caches.target.write_speculative(2)
        caches.draft.write_speculative(2)
        caches.discard_speculative()
        self.assertEqual(caches.target.spec_len, 0)
        self.assertEqual(caches.draft.spec_len, 0)


if __name__ == "__main__":
    unittest.main()

