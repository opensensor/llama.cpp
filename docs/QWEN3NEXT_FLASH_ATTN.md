# Flash Attention for Qwen3Next Linear Attention

This document describes the flash attention optimization for gated delta net (linear attention) layers in Qwen3Next models.

## Overview

Qwen3Next uses a hybrid attention architecture that alternates between:
- **Full attention layers** (every 4th layer)
- **Linear attention layers** (gated delta net / SSM)

The flash attention optimization specifically targets the linear attention layers to improve performance.

## Architecture

### Qwen3Next Attention Pattern

```
Layer 0:  Linear Attention (SSM)
Layer 1:  Linear Attention (SSM)
Layer 2:  Linear Attention (SSM)
Layer 3:  Full Attention
Layer 4:  Linear Attention (SSM)
Layer 5:  Linear Attention (SSM)
Layer 6:  Linear Attention (SSM)
Layer 7:  Full Attention
...
```

### Linear Attention (Gated Delta Net)

The linear attention layers use a recurrent state machine with:
- State dimension: `S_v = 64`
- Number of heads: `H = 32`
- Input projection: Q, K, V, Z
- Convolutional component: 1D convolution with kernel size `d_conv`

## Flash Attention Implementation

### Algorithm

The flash attention for gated delta net combines:
1. **SSM convolution** - Efficient O(n) convolution
2. **Flash attention** - Optimized attention computation O(n²) with memory efficiency

### Key Features

- **Automatic dispatch**: Uses flash attention for `n_tokens > 32`
- **Fallback**: Uses standard gated delta net for state retention (`K > 1`)
- **CUDA optimized**: Custom kernels for S_v = 16, 32, 64, 128
- **KDA support**: Supports both standard and key-dependent activation modes

### Implementation Details

**Files:**
- `ggml/src/ggml-cuda/fattn-gdn.cuh` - Header with kernel templates
- `ggml/src/ggml-cuda/fattn-gdn.cu` - CUDA implementation
- Modified: `ggml/src/ggml-cuda/gated_delta_net.cu` - Dispatch logic

**Kernel Launch Parameters:**
```cpp
dim3 grid_dims(H, n_seqs, (S_v + num_warps - 1) / num_warps);
dim3 block_dims(32, num_warps, 1);
```

Where `num_warps = 4` for optimal occupancy.

## Performance Improvements

### Expected Speedup

| Sequence Length | Standard GDN | Flash GDN | Speedup |
|----------------|--------------|-----------|---------|
| 16             | 1.0x         | 0.9x      | 10% slower (kernel overhead) |
| 32             | 1.0x         | 1.0x      | ~same |
| 64             | 1.0x         | 1.5x      | 50% faster |
| 128            | 1.0x         | 2.0x      | 100% faster |
| 256            | 1.0x         | 2.5x      | 150% faster |
| 512            | 1.0x         | 3.0x      | 200% faster |

### Memory Efficiency

Flash attention reduces memory bandwidth by:
- **Fusing operations**: Convolution + attention in single kernel
- **Register caching**: Q, K cached in registers
- **Reduced global memory accesses**: State maintained in shared memory

## Usage

### Command Line

```bash
# Standard usage (flash attention auto-enabled)
./main -m qwen3-coder-next.gguf -n 512

# With GPU layers
./main -m qwen3-coder-next.gguf -n 512 --n-gpu-layers 80

# With specific batch sizes
./main -m qwen3-coder-next.gguf -n 512 --n-batch 512 --n-ubatch 256
```

### Python API

```python
from llama_cpp import Llama

# Flash attention auto-enabled for linear attention layers
llm = Llama(
    model_path="qwen3-coder-next.gguf",
    n_gpu_layers=80,
)

output = llm("Hello, how are you?", max_tokens=128)
```

## Testing

### Unit Tests

```bash
# Build and run C++ tests
cd build
make test-fattn-gdn
./tests/test-fattn-gdn
```

### Integration Tests

```bash
# Run Python integration tests
python tests/python/test_qwen3next_fattn.py
```

### Test Coverage

1. **Basic functionality** - Verifies graph construction and execution
2. **Correctness** - Compares with CPU reference implementation
3. **Sequence lengths** - Tests various sequence lengths (8-256)
4. **KDA mode** - Tests key-dependent activation
5. **State retention** - Tests with multiple state snapshots
6. **Performance** - Measures throughput for large sequences

## Implementation Notes

### When Flash Attention is Used

Flash attention is automatically selected when:
- `n_tokens > 32` (large enough to benefit from optimization)
- `K == 1` (no state retention needed)

Otherwise, falls back to standard gated delta net.

### State Management

Flash attention implementation maintains recurrent states in:
- **Shared memory**: For fast access during computation
- **Global memory**: For state snapshots when `K > 1`

### CUDA Optimizations

1. **Warp-level reductions**: Efficient parallel reductions
2. **Register tiling**: Q, K cached in registers
3. **Coalesced memory access**: Optimal memory patterns
4. **Kernel fusion**: Combined convolution and attention

## Comparison with Transformers

| Feature | Transformers | llama.cpp (with flash GDN) |
|---------|--------------|----------------------------|
| Flash Attention | Yes | Yes |
| Linear Attention | Limited | Full GDN support |
| Automatic Optimization | Yes | Yes (n_tokens > 32) |
| State Retention | Yes | Yes (K snapshots) |
| GPU Offload | Yes | Yes |
| Quantization | Yes | Yes |

## Future Improvements

1. **Chunked prefill**: Optimize for very long contexts (>4096 tokens)
2. **Mixed precision**: Support FP16/BF16 input tensors
3. **Multi-GPU**: Distribute layers across multiple GPUs
4. **Dynamic dispatch**: More sophisticated kernel selection
5. **Quantized states**: Compress recurrent states

## References

- [Flash Attention Paper](https://arxiv.org/abs/2305.13245)
- [Gated Delta Net](https://arxiv.org/abs/2402.18941)
- [Qwen3Next Architecture](https://github.com/QwenLM/Qwen3)
