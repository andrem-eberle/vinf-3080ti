# vinf

A from-scratch LLM inference engine built for **one GPU: the NVIDIA GeForce RTX 3080 Ti** (Ampere, `sm_86`, 12 GB).
It runs **Qwen3.8-27B** (the `qwen35` hybrid architecture: 48 gated-delta-rule SSM layers + 16 gated
full-attention layers) directly from llama.cpp / Unsloth **GGUF** files, including models larger than VRAM.

There is no other target hardware. Kernels are compiled for `sm_86` only, and memory planning and tuning
assume a single RTX 3080 Ti (80 SMs, 12 GB). Other GPUs are unsupported and untested.

The engine has been tested mostly with **Qwen3.8-27B** (UD-Q3_K_XL and UD-Q4_K_M GGUF quantizations).

## Benchmarks

Measured on an RTX 3080 Ti (PCIe 4.0 x16, desktop session holding ~1.2 GB VRAM), greedy decoding.
Decode: 120 new tokens, chat prompt *"Write a short paragraph about the history of the Roman Empire."* (`--no-think`).
Prefill: a 1910-token prompt, cold (no prefix cache). Speculative modes produce output identical to plain greedy decoding.

### Qwen3.8-27B UD-Q3_K_XL (12.2 GiB GGUF)

| Context | Mode | KV cache | Prefill tok/s | Decode tok/s | Weights on GPU |
|---:|---|---|---:|---:|---:|
| 2048 | **stream + MTP speculation, k=3 (default)** | fp16 | 206 | **11.66** | 8.81 GiB |
| 2048 | stream, no speculation | fp16 | 265 | 7.55 | 9.35 GiB |
| 32768 | stream + MTP speculation, k=3 | fp16 | 205 | 8.96 | 6.86 GiB |
| 32768 | stream + MTP speculation, k=3 | fp32 | 208 | 7.13 | 4.73 GiB |
| 32768 | stream, no speculation | fp16 | 267 | 4.91 | 7.46 GiB |

MTP accepts 2.53 tokens per pass on this prompt. With MTP, prefill also runs the draft block over the prompt
(~22% slower prefill, much faster decode). Follow-up turns of a conversation resume from the prefix cache and only
process their new tokens.

Earlier build (fp32 KV, 8-token prefill passes), context 2048: DFlash 2 draft k=4 decoded at 10.52 tok/s
(2.83 tokens / pass).

### Qwen3.8-27B UD-Q4_K_M (16.4 GiB GGUF, earlier build: fp32 KV, 8-token prefill passes)

| Mode | Decode tok/s |
|---|---:|
| **stream + MTP speculation, k=3** | **8.10** |
| stream + MTP k=2 | 7.77 |
| fused megakernel, no speculation | 4.71 |
| stream, no speculation | 4.37 |

Prefill: 23-token chat prompt in 1.4 s (batched) vs 9.4 s token-by-token.

### Where the time goes

The 27B model does not fit in 12 GB, so every forward pass streams the non-resident weights over PCIe
(~22 GB/s). Decode speed is therefore set by the bytes that do not fit in VRAM; speculative decoding helps
because one pass verifies several tokens for a single weight transfer.

## Features

- **GGUF loading without dequantization on the host**: raw quant blocks go straight to VRAM; CUDA kernels compute on
  F32, F16, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS.
  Every type is verified bit-exact against llama.cpp's reference dequantizer.
- **Live-VRAM residency planner**: fills free VRAM (after KV cache, SSM state, activations, and a safety margin)
  with whole tensors; the rest is streamed from pinned host memory through a copy-engine prefetch ring.
- **Full qwen35 hybrid decoder on the GPU**: gated-query attention with q/k RMSNorm and partial NeoX RoPE, causal
  conv + gated delta rule SSM layers, SwiGLU MLP, LM head with on-device argmax (only token IDs leave the GPU).
- **Batched multi-token passes**: prefill and speculative verification read each weight once for up to 8 tokens.
- **Speculative decoding (lossless greedy)** with SSM/conv state snapshots and rollback:
  - **MTP**: the model's built-in next-token-prediction block (`blk.64`), no extra download.
  - **DFlash / DFlash 2**: block-diffusion drafters (e.g. `z-lab/Qwen3.8-27B-DFlash2`), including grouped
    dynamic convolutions and the candidate-path selector.
- **Fused persistent megakernel** (optional): one cooperative launch per token across all 80 SMs, int8 `dp4a`
  matvec, dependency counters, copy-engine weight streaming.
- **Qwen BPE tokenizer and chat template** read from the GGUF.

## Requirements

- NVIDIA GeForce RTX 3080 Ti (compute capability 8.6, 12 GB)
- Running a release archive:
  - an NVIDIA driver with CUDA 13 support (tested with 595.71.05). The CUDA runtime is linked into the engine;
    no CUDA toolkit is needed.
  - the Python version named in the archive (`cp312` = Python 3.12). The compiled modules load in no other version;
    for a different one, build your own release with `make PYTHON=<interpreter> release`.
  - the GCC OpenMP runtime `libgomp.so.1` (part of the standard GCC runtime on practically every Linux
    distribution; package `libgomp1` on Debian/Ubuntu).
- Building from source: the CUDA toolkit with `nvcc` (`engine/env.cuda.sh` points at `/usr/local/cuda-13.4`; edit for your install)

## Build

```bash
cd engine
source env.cuda.sh
make PYTHON=python native cpu-qwen cuda-qwen-runtime cuda-qwen-megakernel
```

The compiled extensions only load in the interpreter they were built for. The Makefile defaults to
`PYTHON=python3.12`; pass `PYTHON=<your interpreter>` (as above) and run the engine with that same interpreter.

`make PYTHON=python gpu-check` builds and smoke-runs every CUDA extension; `make PYTHON=python check` runs the test suite.

### Release archive

```bash
cd engine && source env.cuda.sh
make PYTHON=python release
```

This produces `engine/dist/vinf-<version>-rtx3080ti-sm86-<python tag>-linux-x86_64.tar.gz`: the engine with its
compiled modules (CUDA runtime included) and a `vinf-run` launcher. Unpack it anywhere and run it with the
same Python version as the tag (e.g. `cp312`):

```bash
tar xzf vinf-*-linux-x86_64.tar.gz && cd vinf-*-linux-x86_64
./vinf-run --model /path/to/Qwen3.8-27B-UD-Q3_K_XL.gguf --serve
```

`vinf-run` accepts every option shown below.

## Usage

```bash
cd engine
source env.cuda.sh

# Chat (defaults: stream placement, MTP speculation k=3)
PYTHONPATH=src python -m vinf.qwen_cli --model /path/to/Qwen3.8-27B-UD-Q3_K_XL.gguf \
    --chat "Explain in one sentence why the sky is blue." --no-think --max-new-tokens 120

# Raw completion
PYTHONPATH=src python -m vinf.qwen_cli --model /path/to/model.gguf --prompt "The capital of France is"

# Memory / residency plan only
PYTHONPATH=src python -m vinf.qwen_cli --model /path/to/model.gguf --report-only
```

The default model path can also be set with `VINF_QWEN_GGUF`.

Useful options:

| Option | Meaning |
|---|---|
| `--speculative K` | draft K tokens per pass (default 3; `0` = plain greedy) |
| `--drafter mtp\|dflash`, `--dflash DIR` | choose the draft model; DFlash needs a checkpoint directory |
| `--executor per-op\|megakernel` | per-op kernels (default) or one fused launch per token (no speculation) |
| `--max-context N` | KV cache size (default 2048); smaller leaves more VRAM for weights |
| `--kv-dtype f16\|f32` | attention KV cache precision (default f16: half the VRAM of f32) |
| `--prefill-batch N` | prompt tokens per pass (default 256) |
| `--no-think` | skip Qwen's reasoning block in chat mode |
| `--safety-mib N` | VRAM left unallocated (default 256; raise if loading runs out of memory) |
| `--profile` | per-op GPU timing breakdown |
| `--serve` | run the OpenAI-compatible HTTP server (see below) |

Sampling is greedy only for now.

## Serving (OpenAI-compatible API)

```bash
cd engine && source env.cuda.sh
PYTHONPATH=src python -m vinf.qwen_cli --model /path/to/Qwen3.8-27B-UD-Q3_K_XL.gguf --serve
```

The server listens on `127.0.0.1:8000` by default (`--host`, `--port`) and accepts all the engine options
above (speculative decoding, drafter, executor, context size). It implements:

| Endpoint | Notes |
|---|---|
| `GET /health` | `{"status": "loading" \| "ok" \| "error"}` |
| `GET /v1/models` | the loaded model (`--served-model-name` overrides the GGUF name) |
| `POST /v1/chat/completions` | the model's own chat template, tool calling, `max_tokens`, `stop`, `stream` (server-sent events), `stream_options.include_usage` |
| `POST /v1/completions` | raw prompt completion, same streaming support |

The `/v1` prefix is optional (`/chat/completions` works too), so clients can use either
`http://127.0.0.1:8000/v1` or `http://127.0.0.1:8000` as the base URL.

**Tool calling** follows the OpenAI format: `tools` (and `tool_choice: "none"`) in the request, `tool_calls` with
`finish_reason: "tool_calls"` in the response (streamed as `tool_calls` deltas), and assistant `tool_calls` /
`tool` messages in the history. Prompts are rendered by an exact port of the GGUF's chat template.

Agentic clients (OpenCode and similar) send long system prompts with many tool definitions; start the server with a
larger context, e.g. `--max-context 32768`. `max_tokens` larger than the room left in the context is clipped.
Prompt processing runs on tensor cores in 256-token passes (~230 tokens/s cold on Q3_K_XL at 32k context), and a
prefix cache (`--prefix-cache-mib`, default 8192) resumes later turns of a conversation from the previous prompt,
so follow-up requests only process the new tokens (a 10k-token turn: ~1.5 s instead of minutes).

Thinking is on by default as in Qwen (the reasoning is returned in `reasoning_content`); turn it off per
request with `"reasoning_effort": "none"` or `"chat_template_kwargs": {"enable_thinking": false}`, or for
the whole server with `--no-think`. Decoding is greedy: `temperature` / `top_p` are accepted and ignored
(`--strict-sampling` rejects them instead). `--api-key KEY` requires `Authorization: Bearer KEY`.
Concurrent requests (several agents on the same port) are decoded together: one pass per step computes the
next token of every generating request, so the weights cross PCIe once for all of them. `--max-seqs N`
(default 4 with `--serve`) sets how many run at once; they share a KV cache pool of `--kv-pool-tokens`
(default `--max-context`). Every request keeps MTP speculation inside the shared pass; when the pool runs out,
the newest request is
swapped to host RAM and resumes later. Each request's output is identical to running it alone.

Total throughput with concurrent requests (Qwen3.8-27B UD-Q3_K_XL, `--max-context 32768`, 100 new tokens each,
including prompt processing):

| `--max-seqs` | Requests at once | Total tok/s |
|---:|---:|---:|
| 4 | 1 / 2 / 4 | 8.6 / 13.2 / 18.1 |
| 16 (`--ssm-dtype f16`) | 1 / 8 / 16 | 6.2 / 30.1 / 39.3 |

More slots hold more per-sequence state in VRAM (fewer weights on the GPU), which slows a lone request; pick
`--max-seqs` for the expected number of agents. `--ssm-dtype f16` halves the SSM state per sequence (~80 MB instead
of ~150 MB). With many sequences, drafts per sequence shrink so a verification pass stays within `--verify-rows`.

```bash
curl http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "Qwen3.8-27B",
  "messages": [{"role": "user", "content": "Explain in one sentence why the sky is blue."}],
  "reasoning_effort": "none",
  "max_tokens": 80
}'
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
reply = client.chat.completions.create(
    model="Qwen3.8-27B",
    messages=[{"role": "user", "content": "Count from 1 to 5."}],
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
print(reply.choices[0].message.content)
```

## Repository layout

```
engine/
  src/vinf/            Python package: GGUF parser and dequantizers, tokenizer, qwen35 executors,
                       residency planner, speculative decoding, DFlash, CLI (qwen_cli.py)
  csrc/megakernel/     CUDA: per-op runtime (qwen_runtime.cu), fused megakernel (qwen_megakernel.cu)
  csrc/common/         RTX 3080 Ti config, quant decoders, IQ codebooks, megakernel ABI
  tests/               unittest suite (synthetic fixtures + optional real-model checks)
```

## Tests

```bash
cd engine && source env.cuda.sh
PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py'
```

Tests that need the real model use `VINF_QWEN_GGUF` and are skipped when the file is absent.

## Status and roadmap

Working: single-sequence greedy decoding of Qwen3.8-27B GGUF models on the RTX 3080 Ti with speculative decoding,
served from the CLI or an OpenAI-compatible HTTP API.
Next:

- Temperature / top-k / top-p sampling with lossless speculative sampling
- Smaller DFlash 2 draft footprint (4-bit draft weights, cheaper SSM snapshots)
- Multi-token passes inside the fused megakernel

## Acknowledgements

- IQ codebooks and quantization layouts follow [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) (MIT).
- DFlash 2 drafter architecture follows [z-lab/dflash](https://github.com/z-lab/dflash).
- Models: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), GGUF quantizations by Unsloth.
