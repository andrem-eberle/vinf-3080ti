from __future__ import annotations

import vinf
from vinf.models.metadata import ModelArchitecture, ModelMetadata
from vinf.native import argmax_float32, native_version
from vinf.runtime.state import RuntimeState
from vinf.sampling import GreedySampler


def main() -> None:
    config = vinf.EngineConfig()
    assert config.target_gpu == "rtx_3080_ti"
    assert native_version() == "vinf-native-0.0.1"
    assert argmax_float32([1.0, 4.0, 2.0]) == 1
    metadata = ModelMetadata(
        architecture=ModelArchitecture.LLAMA,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_kv_heads=1,
        hidden_size=128,
        intermediate_size=256,
        head_dim=64,
        vocab_size=32000,
        max_position_embeddings=128,
        dtype="fp16",
    )
    state = RuntimeState(model=metadata, prompt_tokens=[1, 2])
    state.append_tokens([3])
    assert state.all_tokens == [1, 2, 3]
    assert GreedySampler().sample([0.1, 0.9], vinf.GenerationConfig()).token_id == 1
    print("smoke ok")


if __name__ == "__main__":
    main()
