# vinf Engine

This package is the implementation workspace for the RTX 3080 Ti LLM inference engine.

Current status: Phase 30 complete — real Qwen 27B (qwen35 hybrid) greedy inference on the RTX 3080 Ti GPU executor.

What exists now:

- Python package skeleton under `src/vinf`.
- Minimal CPython C extension exposed as `vinf._native`.
- Smoke tests for import, config creation, and native extension calls.
- A no-network `Makefile` build path for the native smoke extension.
- Public API stubs, engine/generation/speculative config objects, model metadata, runtime state, executor protocols, decode strategy interfaces, sampler interface, metrics, and engine error types.
- CPU-side memory planner, buffer registry, activation/logits buffer plans, target/draft KV cache logical state, and speculative commit/discard bookkeeping.
- Locked first supported target: a tiny JSON `reference_transition_lm` shape used as the correctness oracle for upcoming reference execution.
- Metadata/weight loaders, target/draft tokenizer compatibility checks, shape/dtype assertions, and memory-budget checks.
- Pure-Python reference executor for the locked transition-logits model.
- CPU logits processing, greedy sampling, deterministic RNG sampling, top-k, top-p, and temperature support.
- Dependency-free GGUF parser for v2/v3 metadata and tensor directories.
- Quantized GGUF tensor metadata, byte-size calculation, and mmap slices are supported; actual dequantization/execution is still pending.
- Fixed-width instruction ABI, opcode dataclasses, per-SM schedule builder, NoOp padding, baseline decode schedule, verification schedule, schedule cache, and generated C/CUDA ABI header.
- RTX 3080 Ti `sm_86` CUDA build command generation, Ampere config header, Hopper/Blackwell/TMA gates, `cp.async` helpers, NoOp CUDA smoke source, and runtime hardware checks.
- CUDA environment helper: `source env.cuda.sh`.
- Minimal CUDA NoOp extension bound to Python, launch-dimension checks, instruction/timing buffer allocation, timing readback placeholder, and live `make cuda-run` target.
- CUDA math utility header with FP32/FP16 conversion helpers, reductions, vectorized load/store helpers, and a CUDA math smoke extension with Python reference parity tests.
- CPU reference RMSNorm and standalone CUDA RMSNorm extension with tiny-vector and Qwen hidden-size parity tests.
- CPU reference matvec/GEMV and standalone CUDA row-major matvec extension with projection-shape tests and quantized-access placeholder.
- CPU reference RoPE/KV append and standalone CUDA RoPE/KV extension with cache-position and GQA-layout tests.
- CPU reference one-token attention and standalone CUDA attention extension with GQA/head-selection and short/longer context tests.
- CPU reference gated MLP and standalone CUDA fused gate/up/SiLU/down extension with parity tests.
- CPU reference final RMSNorm/LM head and standalone CUDA LM-head extension emitting logits.
- CPU reference one-layer decode and CUDA-composed one-layer decode with per-op stop-point parity.
- CPU reference full baseline one-token decode over all layers with final logits, schedule, timing slot metadata, and CUDA-composed parity when the standalone CUDA ops are available.
- Target megakernel runtime entrypoint that allocates full instruction/timing buffers and completes one-token baseline decode through the current CUDA-composed target path.
- Reference/fallback prefill scaffold over hidden states, with persistent target KV cache fill, prefill/decode handoff checks, long-prompt validation, and memory-use estimates.
- Baseline decode loop scaffold that connects prefill, decode, CPU sampling, stop conditions, streaming token IDs, metrics, and a reproducible tokens/sec benchmark over hidden-state prompts.
- Isolated speculative sampling math for host-side acceptance/correction, with deterministic tests for the formulas in the speculative decoding design document.
- Decoupled speculative interfaces, CPU sampler, logical KV commit manager, and strategy shell with speculative mode disabled by default.
- Repeat-last heuristic draft runner with tokenizer compatibility checks, logical draft KV tracking/rollback, stored draft tokens/probability rows, and draft cost metrics.
- Draft-path integration tests covering drafter to sampler to KV commit, rejected suffix invisibility, and the speculative disabled baseline boundary.
- Fallback target verifier for speculative decoding that returns `gamma + 1` probability rows and keeps target verification KV speculative until accepted.
- Speculative hybrid loop that connects draft proposal, target verification, sampler, KV commit, metrics, adaptive disable, and a reproducible benchmark.
- Speculative megakernel verification contract: verification globals, `max_gamma`/`num_verify_tokens`, multi-position activation/logit/probability/speculative-KV buffer shapes, causal verify mask, and verification schedule.
- CUDA speculative verifier entrypoint scaffold that transforms per-position logits into probability rows, with Python binding and build/parity tests.
- GPU-sampling contract scaffold with logits processing, FP32 softmax/probability transform, deterministic counter RNG, token-id-only sampling, speculative acceptance/correction, and fixed-seed parity tests.

Current environment limitation:

- `make cuda-smoke` compiles after `source env.cuda.sh`.
- `make cuda-run` launches the NoOp kernel when the CUDA runtime can see the RTX 3080 Ti.
- `make cuda-math` builds and runs the CUDA math helper kernel when the CUDA runtime can see the RTX 3080 Ti.
- `make cuda-rmsnorm` builds and runs the CUDA RMSNorm kernel when the CUDA runtime can see the RTX 3080 Ti.
- `make cuda-matvec` builds and runs the CUDA matvec kernel when the CUDA runtime can see the RTX 3080 Ti.
- `make cuda-rope-kv` builds and runs the CUDA RoPE/KV append kernel when the CUDA runtime can see the RTX 3080 Ti.
- `make cuda-attn` builds and runs the CUDA one-token attention kernel when the CUDA runtime can see the RTX 3080 Ti.
- `make cuda-mlp` builds and runs the CUDA MLP kernel when the CUDA runtime can see the RTX 3080 Ti.
- `make cuda-lm-head` builds and runs the CUDA LM-head kernel when the CUDA runtime can see the RTX 3080 Ti.
- `make gpu-check` runs all live GPU checks that require driver/runtime visibility.

## Running Qwen 27B (Phase 30)

Execution interpreter is `python3.12`. Build once:

```bash
source env.cuda.sh
make native cuda-qwen-runtime
```

Memory/residency plan only (no weight upload):

```bash
PYTHONPATH=src python3.12 -m vinf.qwen_cli --report-only
```

First-token bring-up and greedy generation:

```bash
PYTHONPATH=src python3.12 -m vinf.qwen_cli --prompt "The capital of France is" --max-new-tokens 12
```

```bash
PYTHONPATH=src python3.12 -m vinf.qwen_cli --chat "Explain in one sentence why the sky is blue." --no-think --max-new-tokens 60
```

Expected smoke output (RTX 3080 Ti with ~10.6 GiB free VRAM, `max_context=2048`):

```
GPU-resident weights    9.56 GiB (693 tensors)
streamed per token      4.77 GiB (157 tensors, pinned host)
CPU mmap                0.67 GiB (token_embd.weight)
prompt tokens (5): [760, 6511, 314, 9338, 369]
output: The capital of France is Paris.
...
decode ... (~2.5 tok/s)
```

The chat example answers with a one-sentence Rayleigh-scattering explanation and stops at `<|im_end|>`.

How it runs (`src/vinf/qwen_gpu.py`, `csrc/megakernel/qwen_runtime.cu`):

- Raw GGUF blocks are uploaded without CPU dequantization; CUDA matvec kernels read F32, F16, Q8_0, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL, IQ4_XS, IQ3_S directly.
- The residency planner fills live free VRAM (after reserving KV cache, SSM state, activations, staging, and a safety margin); remaining matrices sit in pinned host memory and are streamed into a device staging buffer on use.
- Every op of the qwen35 hybrid stack runs on the GPU: 48 gated-delta-rule (SSM) layers and 16 gated full-attention layers, MLPs, final norm, LM head, argmax. Only the token embedding row (CPU-dequantized) goes up and only the argmax token id comes back.
- SSM value heads use the tiled order produced by llama.cpp GGUF conversion (`--head-order grouped` for HF-ordered weights).
- Errors (insufficient VRAM, unsupported quant type, bad config) are reported as `error: ...` with exit code 2.

What does not exist yet:

- A fused CUDA target megakernel that executes the full one-token decode schedule inside a single kernel launch.
- A real multi-position speculative target megakernel verifier to replace the fallback verifier.
- Batched (multi-token) prefill; prompts are prefilled one token at a time through the decode path.
- Sampling other than greedy in the qwen35 GPU path.
- Neural draft executor backed by a real same-tokenizer draft model.

The vendored `../Megakernels` repository remains separate. Engine code should wrap or include it from this package rather than modifying demo files in place.

Run the current smoke check with:

```bash
make check
```
