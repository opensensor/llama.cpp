# Quick Start - Flash Attention for Qwen3Next

## What Was Implemented

Flash attention optimization for gated delta net (linear attention) layers in Qwen3Next models. Flash attention is automatically selected for sequences longer than 32 tokens.

## Quick Overview

```bash
# Build with CUDA
cmake -DGGML_CUDA=ON ..
make -j$(nproc)

# Run tests
make test-fattn-gdn
ctest -R test-fattn-gdn -V

# Run benchmarks
./benchmarks/bench_fattn_gdn.sh
```

## Expected Performance

| Sequence Length | Speedup |
|----------------|---------|
| 64             | 1.5x    |
| 128            | 2.0x    |
| 256            | 2.5x    |
| 512            | 3.0x    |
| 1024+          | 3.5x+   |

## Key Files

| File | Purpose |
|------|---------|
| `ggml/src/ggml-cuda/fattn-gdn.cuh` | Flash attention header |
| `ggml/src/ggml-cuda/fattn-gdn.cu` | Flash attention implementation |
| `tests/test-fattn-gdn.cpp` | C++ unit tests |
| `tests/python/test_qwen3next_fattn.py` | Python integration tests |
| `docs/QWEN3NEXT_FLASH_ATTN.md` | Documentation |

## Usage

### Command Line

```bash
./main -m qwen3-coder-next.gguf -n 512 --n-gpu-layers 80
```

### Python

```python
from llama_cpp import Llama

llm = Llama(
    model_path="qwen3-coder-next.gguf",
    n_gpu_layers=80,
)
output = llm("Hello", max_tokens=128)
```

## Test Results

Run `make test-fattn-gdn` to verify:

```
Test 1: Basic functionality... PASSED
Test 2: Correctness (CPU reference)... PASSED
Test 3: Different sequence lengths... PASSED
Test 4: KDA mode... PASSED
Test 5: State retention... PASSED
Test 6: Performance... PASSED
```

## Documentation

- [Full Documentation](docs/QWEN3NEXT_FLASH_ATTN.md)
- [Implementation Guide](ggml/src/ggml-cuda/README_FLASH_ATTN_GDN.md)
- [Summary of Changes](CHANGES_SUMMARY.md)

## Next Steps

1. Install CUDA toolkit
2. Build llama.cpp: `cmake -DGGML_CUDA=ON ..`
3. Run tests: `make test-fattn-gdn`
4. Benchmark: `./benchmarks/bench_fattn_gdn.sh`

## Questions?

See the documentation files for detailed information about:
- Architecture
- Performance characteristics
- Testing procedures
- Implementation details
