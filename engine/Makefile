PYTHON ?= python3.12
CC ?= cc
EXT_SUFFIX := $(shell $(PYTHON)-config --extension-suffix)
PY_INCLUDES := $(shell $(PYTHON)-config --includes)
NATIVE_OUT := src/vinf/_native$(EXT_SUFFIX)

.PHONY: native cpu-qwen release cuda-smoke cuda-check cuda-run cuda-math cuda-gguf-upload cuda-qwen-runtime cuda-qwen-megakernel gpu-check check clean

native: $(NATIVE_OUT)

$(NATIVE_OUT): csrc/native.c
	$(CC) -O3 -fPIC -shared $(PY_INCLUDES) $< -o $@

CPU_QWEN_OUT := src/vinf/_cpu_qwen$(EXT_SUFFIX)

# CPU layers for weights that do not fit in VRAM (AVX2/FMA/F16C, OpenMP).
cpu-qwen: $(CPU_QWEN_OUT)

$(CPU_QWEN_OUT): csrc/cpu/qwen_cpu.c csrc/common/qwen_iq_tables.h
	$(CC) -O3 -march=native -fopenmp -fPIC -shared $(PY_INCLUDES) -Icsrc/common $< -o $@

check: native cpu-qwen
	PYTHONPATH=src $(PYTHON) -m compileall -q src tests
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -p 'test_*.py'
	PYTHONPATH=src $(PYTHON) tests/smoke.py

cuda-smoke:
	mkdir -p build
	PYTHONPATH=src $(PYTHON) -c "from vinf.cuda.build import cuda_smoke_compile_command; import subprocess; cmd=cuda_smoke_compile_command('.'); print(' '.join(cmd)); subprocess.check_call(cmd)"

cuda-run: cuda-smoke
	PYTHONPATH=src $(PYTHON) -c "from vinf.cuda.noop import run_and_validate_noop_smoke; print(run_and_validate_noop_smoke())"

cuda-math:
	mkdir -p build
	PYTHONPATH=src $(PYTHON) -c "from vinf.cuda.build import cuda_extension_compile_command; import subprocess; cmd=cuda_extension_compile_command('.', source='csrc/megakernel/math_smoke.cu', output='src/vinf/_cuda_math.so'); print(' '.join(cmd)); subprocess.check_call(cmd)"
	PYTHONPATH=src $(PYTHON) -c "from vinf.cuda.math import run_math_smoke; print(run_math_smoke([1.0, 2.0, 3.0, 4.0, 5.0]))"

gpu-check: cuda-check cuda-run cuda-math cuda-rmsnorm cuda-matvec cuda-rope-kv cuda-attn cuda-mlp cuda-lm-head cuda-spec-verify cuda-gguf-upload cuda-qwen-runtime cuda-qwen-megakernel

cuda-matvec:
	mkdir -p build
	PYTHONPATH=src $(PYTHON) -c "from vinf.cuda.build import cuda_extension_compile_command; import subprocess; cmd=cuda_extension_compile_command('.', source='csrc/megakernel/matvec.cu', output='src/vinf/_cuda_matvec.so'); print(' '.join(cmd)); subprocess.check_call(cmd)"
	PYTHONPATH=src $(PYTHON) -c "from vinf.cuda.matvec import matvec_cuda; print(matvec_cuda([1.0, 2.0, 3.0], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0], 2, 3))"

cuda-rope-kv:
	mkdir -p build
	PYTHONPATH=src $(PYTHON) -c "from vinf.cuda.build import cuda_extension_compile_command; import subprocess; cmd=cuda_extension_compile_command('.', source='csrc/megakernel/rope_kv.cu', output='src/vinf/_cuda_rope_kv.so'); print(' '.join(cmd)); subprocess.check_call(cmd)"
	PYTHONPATH=src $(PYTHON) -c "from vinf.cuda.rope_kv import rope_kv_cuda; print(rope_kv_cuda([1.0, 2.0, 3.0, 4.0], [1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0], num_heads=2, max_seq=3, head_idx=1, position=2))"

cuda-attn:
	mkdir -p build
	PYTHONPATH=src $(PYTHON) -c "from vinf.cuda.build import cuda_extension_compile_command; import subprocess; cmd=cuda_extension_compile_command('.', source='csrc/megakernel/attention.cu', output='src/vinf/_cuda_attention.so'); print(' '.join(cmd)); subprocess.check_call(cmd)"
	PYTHONPATH=src $(PYTHON) -c "from vinf.cuda.attention import attention_cuda; print(attention_cuda([1.0, 0.0], [1.0, 0.0, 0.0, 1.0], [10.0, 0.0, 0.0, 20.0], kv_head_idx=0, seq_len=2, num_kv_heads=1, max_seq=2, head_dim=2, scale=1.0))"

cuda-mlp:
	mkdir -p build
	PYTHONPATH=src $(PYTHON) -c "from vinf.cuda.build import cuda_extension_compile_command; import subprocess; cmd=cuda_extension_compile_command('.', source='csrc/megakernel/mlp.cu', output='src/vinf/_cuda_mlp.so'); print(' '.join(cmd)); subprocess.check_call(cmd)"
	PYTHONPATH=src $(PYTHON) -c "from vinf.cuda.mlp import mlp_cuda; print(mlp_cuda([1.0, 2.0], [1.0,0.0, 0.0,1.0], [1.0,1.0, 1.0,1.0], [1.0,0.0, 0.0,1.0], 2, 2))"

cuda-lm-head:
	mkdir -p build
	PYTHONPATH=src $(PYTHON) -c "from vinf.cuda.build import cuda_extension_compile_command; import subprocess; cmd=cuda_extension_compile_command('.', source='csrc/megakernel/lm_head.cu', output='src/vinf/_cuda_lm_head.so'); print(' '.join(cmd)); subprocess.check_call(cmd)"
	PYTHONPATH=src $(PYTHON) -c "from vinf.cuda.lm_head import lm_head_cuda; print(lm_head_cuda([1.0,2.0], [1.0,1.0], [1.0,0.0, 0.0,1.0, 1.0,1.0], 2, 3))"

cuda-spec-verify:
	mkdir -p build
	PYTHONPATH=src $(PYTHON) -c "from vinf.cuda.build import cuda_extension_compile_command; import subprocess; cmd=cuda_extension_compile_command('.', source='csrc/megakernel/spec_verify.cu', output='src/vinf/_cuda_spec_verify.so'); print(' '.join(cmd)); subprocess.check_call(cmd)"
	PYTHONPATH=src $(PYTHON) -c "from vinf.cuda.spec_verify import spec_verify_cuda; print(spec_verify_cuda([[0.0, 1.0], [2.0, 0.0]], num_verify_tokens=2, vocab_size=2))"

cuda-gguf-upload:
	mkdir -p build
	PYTHONPATH=src $(PYTHON) -c "from vinf.cuda.build import cuda_extension_compile_command; import subprocess; cmd=cuda_extension_compile_command('.', source='csrc/megakernel/gguf_upload.cu', output='src/vinf/_cuda_gguf_upload.so'); print(' '.join(cmd)); subprocess.check_call(cmd)"
	PYTHONPATH=src $(PYTHON) -c "from vinf.cuda.gguf_upload import upload_checksum_cuda; print(upload_checksum_cuda(bytes(range(16))))"

cuda-qwen-runtime:
	mkdir -p build
	PYTHONPATH=src $(PYTHON) -c "from vinf.cuda.build import cuda_extension_compile_command; import subprocess; cmd=cuda_extension_compile_command('.', source='csrc/megakernel/qwen_runtime.cu', output='src/vinf/_cuda_qwen_runtime.so', extra_flags=('-lcuda',)); print(' '.join(cmd)); subprocess.check_call(cmd)"
	PYTHONPATH=src $(PYTHON) -c "from vinf.cuda.qwen_runtime import CudaWeightRuntime; print(CudaWeightRuntime().mem_info())"

# -dlcm=cg: global loads bypass L1 so data written by other SMs during the launch is coherent.
cuda-qwen-megakernel:
	mkdir -p build
	PYTHONPATH=src $(PYTHON) -c "from vinf.runtime.instructions import qwen_mk_abi_header; open('csrc/common/qwen_mk_abi.h', 'w').write(qwen_mk_abi_header())"
	PYTHONPATH=src $(PYTHON) -c "from vinf.cuda.build import cuda_extension_compile_command; import subprocess; cmd=cuda_extension_compile_command('.', source='csrc/megakernel/qwen_megakernel.cu', output='src/vinf/_cuda_qwen_megakernel.so', extra_flags=('-Xptxas', '-dlcm=cg', '-lcuda')); print(' '.join(cmd)); subprocess.check_call(cmd)"
	PYTHONPATH=src $(PYTHON) -c "from vinf.cuda.qwen_runtime import _iq3_s_grid_bytes; from vinf import _cuda_qwen_megakernel as m; print(m.Megakernel(_iq3_s_grid_bytes()).device_info())"

cuda-rmsnorm:
	mkdir -p build
	PYTHONPATH=src $(PYTHON) -c "from vinf.cuda.build import cuda_extension_compile_command; import subprocess; cmd=cuda_extension_compile_command('.', source='csrc/megakernel/rmsnorm.cu', output='src/vinf/_cuda_rmsnorm.so'); print(' '.join(cmd)); subprocess.check_call(cmd)"
	PYTHONPATH=src $(PYTHON) -c "from vinf.cuda.rmsnorm import rmsnorm_cuda; print(rmsnorm_cuda([1.0, 2.0, 3.0, 4.0], [1.0, 1.0, 1.0, 1.0]))"

cuda-check:
	PYTHONPATH=src $(PYTHON) -c "from vinf.cuda.hardware import check_rtx_3080_ti; print(check_rtx_3080_ti())"

clean:
	rm -rf build dist src/vinf/_native*.so src/vinf/_cpu_qwen*.so src/vinf/_cuda_noop*.so src/vinf/_cuda_math*.so src/vinf/_cuda_rmsnorm*.so src/vinf/_cuda_matvec*.so src/vinf/_cuda_rope_kv*.so src/vinf/_cuda_attention*.so src/vinf/_cuda_mlp*.so src/vinf/_cuda_lm_head*.so src/vinf/_cuda_spec_verify*.so src/vinf/_cuda_gguf_upload*.so src/vinf/_cuda_qwen_runtime*.so src/vinf/_cuda_qwen_megakernel*.so

# ---- release ---------------------------------------------------------------------------------
# Self-contained archive for the RTX 3080 Ti: the CUDA runtime is linked statically, so running it
# needs only the NVIDIA driver and the same Python version the modules were built for.
VERSION := $(shell sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml)
PYTAG := $(shell $(PYTHON) -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
RELEASE := vinf-$(VERSION)-rtx3080ti-sm86-$(PYTAG)-linux-x86_64
RELEASE_MODULES := src/vinf/_native$(EXT_SUFFIX) src/vinf/_cpu_qwen$(EXT_SUFFIX) src/vinf/_cuda_qwen_runtime.so src/vinf/_cuda_qwen_megakernel.so

release:
	rm -f $(RELEASE_MODULES)
	$(MAKE) PYTHON=$(PYTHON) native cpu-qwen cuda-qwen-runtime cuda-qwen-megakernel
	@for mod in src/vinf/_cuda_qwen_runtime.so src/vinf/_cuda_qwen_megakernel.so; do \
		if ldd $$mod | grep -q libcudart; then echo "error: $$mod links libcudart dynamically"; exit 1; fi; \
	done
	rm -rf dist/$(RELEASE) dist/$(RELEASE).tar.gz
	$(PYTHON) -c "import shutil; shutil.copytree('src/vinf', 'dist/$(RELEASE)/vinf', ignore=shutil.ignore_patterns('__pycache__', '*.so'))"
	cp $(RELEASE_MODULES) dist/$(RELEASE)/vinf/
	cp ../README.md dist/$(RELEASE)/README.md
	printf '#!/bin/sh\n# vinf launcher: run the engine from this directory (needs $(PYTHON) and the NVIDIA driver).\nDIR=$$(dirname "$$(readlink -f "$$0")")\nPYTHONPATH="$$DIR$${PYTHONPATH:+:$$PYTHONPATH}" exec $(PYTHON) -m vinf.qwen_cli "$$@"\n' > dist/$(RELEASE)/vinf-run
	chmod +x dist/$(RELEASE)/vinf-run
	tar -C dist -czf dist/$(RELEASE).tar.gz $(RELEASE)
	@echo "release: dist/$(RELEASE).tar.gz"
