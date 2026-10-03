from __future__ import annotations

import dataclasses
import json
import random
import struct
import tempfile
import unittest
from pathlib import Path

from tests.test_phase30_qwen_gpu import gpu_fixture, gpu_metadata
from vinf.dflash import DFlashCheckpoint, DFlashConfig, dflash_reference_draft, dflash_tensor_names
from vinf.errors import ExecutorUnavailableError, UnsupportedModelError

PROMPT = [3, 17, 5, 29, 11, 2, 7, 19, 23, 1]

# Published z-lab/Qwen3.8-27B-DFlash2 config (architecture fields).
PUBLISHED_CONFIG = {
    "architectures": ["DFlash2DraftModel"],
    "dflash_config": {"block_size": 8, "conv_group_size": 16, "conv_kernel_size": 2, "mask_token_id": 248070,
                      "selector_rank": 256, "selector_top_k": 16, "target_layer_ids": [5, 19, 33, 47, 61]},
    "head_dim": 128, "hidden_size": 5120, "intermediate_size": 17408, "num_attention_heads": 32,
    "num_hidden_layers": 5, "num_key_value_heads": 8, "rms_norm_eps": 1e-06,
    "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"}, "sliding_window": 2048,
    "use_sliding_window": True, "vocab_size": 248320,
}


def tiny_config(**overrides) -> dict:
    cfg = {
        "architectures": ["DFlash2DraftModel"],
        "dflash_config": {"block_size": 4, "mask_token_id": 39, "target_layer_ids": [1, 3],
                          "conv_kernel_size": 2, "conv_group_size": 16, "selector_top_k": 4, "selector_rank": 8},
        "head_dim": 32, "hidden_size": 64, "intermediate_size": 96, "num_attention_heads": 2,
        "num_hidden_layers": 2, "num_key_value_heads": 1, "rms_norm_eps": 1e-6,
        "rope_parameters": {"rope_theta": 10000.0}, "sliding_window": 6, "use_sliding_window": True,
        "vocab_size": 40,
    }
    cfg.update(overrides)
    return cfg


def bf16_bytes(values: list[float]) -> bytes:
    return b"".join(struct.pack("<H", struct.unpack("<I", struct.pack("<f", v))[0] >> 16) for v in values)


def write_checkpoint(cfg: dict, extra: dict[str, tuple[int, ...]] | None = None, prefix: str = "", seed: int = 5) -> Path:
    config = DFlashConfig.from_dict(cfg)
    rng = random.Random(seed)
    mats, vecs = dflash_tensor_names(config)
    c = config
    shapes = {"fc.weight": (c.hidden_size, len(c.target_layer_ids) * c.hidden_size)}
    for i in range(c.num_layers):
        p = f"layers.{i}."
        shapes.update({
            p + "self_attn.q_proj.weight": (c.num_heads * c.head_dim, c.hidden_size),
            p + "self_attn.k_proj.weight": (c.num_kv_heads * c.head_dim, c.hidden_size),
            p + "self_attn.v_proj.weight": (c.num_kv_heads * c.head_dim, c.hidden_size),
            p + "self_attn.o_proj.weight": (c.hidden_size, c.num_heads * c.head_dim),
            p + "mlp.gate_proj.weight": (c.intermediate_size, c.hidden_size),
            p + "mlp.up_proj.weight": (c.intermediate_size, c.hidden_size),
            p + "mlp.down_proj.weight": (c.hidden_size, c.intermediate_size),
        })
    for name in vecs:
        dim = c.head_dim if name.endswith(("q_norm.weight", "k_norm.weight")) else c.hidden_size
        shapes[name] = (dim,)
    if c.version == 2:
        groups = c.hidden_size // c.conv_group_size
        for i in range(c.num_layers):
            for conv in ("attention_conv", "mlp_conv"):
                shapes[f"layers.{i}.{conv}.base_kernel"] = (2, c.conv_kernel_size, c.hidden_size)
                shapes[f"layers.{i}.{conv}.kernel_projection.weight"] = (2 * c.conv_kernel_size * groups, c.hidden_size)
        shapes["candidate_selector.hidden_projection.weight"] = (c.selector_rank, c.hidden_size)
        shapes["candidate_selector.predecessor_codebook"] = (c.vocab_size, c.selector_rank)
        shapes["candidate_selector.successor_codebook"] = (c.vocab_size, c.selector_rank)
    shapes.update(extra or {})
    header, payload = {}, bytearray()
    for name, shape in shapes.items():
        n = 1
        for d in shape:
            n *= d
        if name.endswith("norm.weight"):
            vals = [rng.uniform(0.7, 1.3) for _ in range(n)]
        elif name.endswith("base_kernel"):
            vals = [rng.uniform(0.3, 0.7) for _ in range(n)]
        elif "codebook" in name:
            vals = [rng.uniform(-1.5, 1.5) for _ in range(n)]
        else:
            vals = [rng.uniform(-0.3, 0.3) for _ in range(n)]
        data = bf16_bytes(vals)
        header[prefix + name] = {"dtype": "BF16", "shape": list(shape), "data_offsets": [len(payload), len(payload) + len(data)]}
        payload.extend(data)
    directory = Path(tempfile.mkdtemp(prefix="dflash_"))
    blob = json.dumps(header).encode()
    (directory / "model.safetensors").write_bytes(struct.pack("<Q", len(blob)) + blob + bytes(payload))
    (directory / "config.json").write_text(json.dumps(cfg))
    return directory


def target(test, **kwargs):
    meta = dataclasses.replace(gpu_metadata(), max_position_embeddings=32)
    gguf = gpu_fixture(meta)
    try:
        from vinf.qwen_gpu import QwenGpuExecutor

        ex = QwenGpuExecutor(gguf, meta, max_context=32, capture_layers=(1, 3), **kwargs)
    except (ExecutorUnavailableError, RuntimeError) as exc:
        test.skipTest(str(exc))
    return meta, gguf, ex


class Phase31fDFlashConfigTests(unittest.TestCase):
    def test_published_dflash2_config_parses(self) -> None:
        c = DFlashConfig.from_dict(PUBLISHED_CONFIG)
        self.assertEqual(c.version, 2)
        self.assertEqual(c.target_layer_ids, (5, 19, 33, 47, 61))
        self.assertEqual((c.block_size, c.sliding_window, c.mask_token_id), (8, 2048, 248070))
        self.assertEqual((c.num_heads, c.num_kv_heads, c.head_dim, c.rope_theta), (32, 8, 128, 1e7))

    def test_checkpoint_loading_prefix_and_unimplemented_tensors(self) -> None:
        ck = DFlashCheckpoint(write_checkpoint(tiny_config(), prefix="model."))
        self.assertEqual(ck.prefix, "model.")
        self.assertEqual(len(ck.floats("norm.weight")), 64)
        extra = {"layers.0.conv.weight": (64, 2), "selector.proj.weight": (8, 64)}
        path = write_checkpoint(tiny_config(), extra=extra)
        with self.assertRaises(UnsupportedModelError) as ctx:
            DFlashCheckpoint(path)
        self.assertIn("layers.0.conv.weight", str(ctx.exception))
        self.assertEqual(DFlashCheckpoint(path, allow_unimplemented=True).unimplemented,
                         ["layers.0.conv.weight", "selector.proj.weight"])

    def test_shape_mismatch_fails_cleanly(self) -> None:
        path = write_checkpoint(tiny_config())
        (path / "config.json").write_text(json.dumps(tiny_config(intermediate_size=64)))
        with self.assertRaises(UnsupportedModelError):
            DFlashCheckpoint(path)


class Phase31fDFlashDrafterTests(unittest.TestCase):
    def test_gpu_block_draft_matches_reference(self) -> None:
        from vinf.dflash import DFlashDrafter

        ck = DFlashCheckpoint(write_checkpoint(tiny_config()))
        meta, gguf, ex = target(self)
        drafter = DFlashDrafter(ex, ck, quantize=False)
        ex.reset()
        ex.forward_tokens(PROMPT[:6])
        feats = ex.rt.read_floats("cap_feat", 6 * 2 * 64)
        features = [feats[p * 128:(p + 1) * 128] for p in range(6)]
        drafter.add_context(0, 6)
        drafts = drafter.draft_block(PROMPT[6], 6, 3)  # block of 4 at positions 6..9, window 6
        gpu_rows = ex.rt.read_floats("dfl_out", 3 * 64)
        ref_drafts, ref_rows = dflash_reference_draft(ck, gguf, features, PROMPT[6], 6, 3)
        for a, b in zip(gpu_rows, [v for row in ref_rows for v in row]):
            self.assertLessEqual(abs(a - b), 2e-4 * max(1.0, abs(b)))
        self.assertEqual(drafts, ref_drafts)

    def test_dflash1_block_draft_matches_reference(self) -> None:
        from vinf.dflash import DFlashDrafter

        cfg = tiny_config(architectures=["DFlashDraftModel"])
        cfg["dflash_config"] = {"block_size": 4, "mask_token_id": 39, "target_layer_ids": [1, 3]}
        ck = DFlashCheckpoint(write_checkpoint(cfg))
        self.assertEqual(ck.config.version, 1)
        meta, gguf, ex = target(self)
        drafter = DFlashDrafter(ex, ck, quantize=False)
        ex.reset()
        ex.forward_tokens(PROMPT[:7])
        feats = ex.rt.read_floats("cap_feat", 7 * 128)
        drafter.add_context(0, 7)
        drafts = drafter.draft_block(PROMPT[7], 7, 2)
        ref, _ = dflash_reference_draft(ck, gguf, [feats[p * 128:(p + 1) * 128] for p in range(7)], PROMPT[7], 7, 2)
        self.assertEqual(drafts, ref)

    def test_capture_matches_layer_outputs(self) -> None:
        meta, gguf, ex = target(self)
        plain = target(self)[2]
        ex.reset()
        ex.forward_tokens(PROMPT[:3])
        feats = ex.rt.read_floats("cap_feat", 3 * 2 * 64)
        # Layer 3 is the last decoder layer: its output is the final pre-norm hidden row.
        plain.reset()
        plain.forward_tokens(PROMPT[:3])
        last = plain.rt.read_floats("h", 3 * 64)
        for t in range(3):
            self.assertEqual(feats[t * 128 + 64:(t + 1) * 128], last[t * 64:(t + 1) * 64])

    def test_dflash_speculative_output_equals_greedy(self) -> None:
        from vinf.dflash import DFlashDrafter
        from vinf.qwen_speculative import QwenSpeculativeDecoder

        ck = DFlashCheckpoint(write_checkpoint(tiny_config()))
        meta, gguf, plain = target(self)
        expected = plain.generate_greedy(PROMPT, 12)[0]
        for quantize in (False, True):
            for k in (1, 2, 3):
                meta, gguf, ex = target(self, snapshot_tokens=k + 1)
                dec = QwenSpeculativeDecoder(ex, k, drafter=DFlashDrafter(ex, ck, quantize=quantize))
                tokens, stats = dec.generate(PROMPT, 12)
                self.assertEqual(tokens, expected, (quantize, k))
                self.assertEqual(sum(stats.accepted_histogram.values()), stats.steps)

    def test_drafter_requires_matching_capture_layers(self) -> None:
        from vinf.dflash import DFlashDrafter
        from vinf.errors import ConfigurationError

        ck = DFlashCheckpoint(write_checkpoint(tiny_config()))
        meta = dataclasses.replace(gpu_metadata(), max_position_embeddings=32)
        try:
            from vinf.qwen_gpu import QwenGpuExecutor

            ex = QwenGpuExecutor(gpu_fixture(meta), meta, max_context=32)
        except (ExecutorUnavailableError, RuntimeError) as exc:
            self.skipTest(str(exc))
        with self.assertRaises(ConfigurationError):
            DFlashDrafter(ex, ck)


if __name__ == "__main__":
    unittest.main()
