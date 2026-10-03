from __future__ import annotations

from time import perf_counter

from vinf.baseline_decode import BaselineDecodeConfig, BaselineTargetWeights
from vinf.baseline_loop import BaselineEngineLoop
from vinf.config import GenerationConfig
from vinf.models.metadata import ModelArchitecture, ModelMetadata
from vinf.one_layer import OneLayerWeights
from vinf.sampling import GreedySampler


def main() -> None:
    loop, rope_cos, rope_sin = fixture(max_seq=32, max_new_tokens=16)
    prompt = prompt_states(8)
    started = perf_counter()
    result = loop.generate_from_hidden_states(prompt, rope_cos=rope_cos, rope_sin=rope_sin)
    elapsed = perf_counter() - started
    tokens = len(result.decode_result.tokens)
    tokens_per_second = tokens / elapsed if elapsed > 0 else 0.0
    print(f"generated_tokens={tokens}")
    print(f"target_calls={result.metrics.target_calls}")
    print(f"elapsed_seconds={elapsed:.6f}")
    print(f"tokens_per_second={tokens_per_second:.2f}")
    print(f"stop_reason={result.decode_result.stop_reason}")


def fixture(max_seq: int, max_new_tokens: int):
    metadata = ModelMetadata(
        architecture=ModelArchitecture.LLAMA,
        num_hidden_layers=2,
        num_attention_heads=1,
        num_kv_heads=1,
        hidden_size=4,
        intermediate_size=8,
        head_dim=4,
        vocab_size=5,
        max_position_embeddings=max_seq,
        dtype="fp16",
    )
    identity4 = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]
    gate_up = [((i % 5) - 2) / 5.0 for i in range(8 * 4)]
    down = [((i % 7) - 3) / 7.0 for i in range(4 * 8)]
    weights = BaselineTargetWeights(
        layers=tuple(
            OneLayerWeights(
                attn_norm=[1.0 + layer_idx * 0.05, 1.1, 0.9, 1.0],
                q_proj=identity4,
                k_proj=identity4,
                v_proj=identity4,
                o_proj=identity4,
                mlp_norm=[1.0, 1.0 + layer_idx * 0.05, 1.0, 1.0],
                gate_proj=gate_up,
                up_proj=list(reversed(gate_up)),
                down_proj=down,
            )
            for layer_idx in range(metadata.num_hidden_layers)
        ),
        final_norm=[1.0, 1.0, 0.95, 1.05],
        lm_head=[((i % 9) - 4) / 9.0 for i in range(metadata.vocab_size * metadata.hidden_size)],
    )
    loop = BaselineEngineLoop(
        weights=weights,
        config=BaselineDecodeConfig(metadata=metadata, num_sms=4, block_size=2),
        generation=GenerationConfig(max_new_tokens=max_new_tokens),
        sampler=GreedySampler(),
    )
    rope_cos = [[1.0, 1.0, 0.0, 0.0] for _ in range(max_seq)]
    rope_sin = [[0.0, 0.0, 1.0, 1.0] for _ in range(max_seq)]
    return loop, rope_cos, rope_sin


def prompt_states(count: int) -> list[list[float]]:
    return [
        [0.25 + idx * 0.1, -0.5 + idx * 0.05, 0.75 - idx * 0.02, 1.0 + idx * 0.03]
        for idx in range(count)
    ]


if __name__ == "__main__":
    main()
