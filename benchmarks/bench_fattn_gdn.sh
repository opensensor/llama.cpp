#!/bin/bash
# Benchmark script for flash attention in gated delta net
# Tests various sequence lengths and compares performance

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}Flash Attention GDN Benchmark${NC}"
echo -e "${BLUE}======================================${NC}"
echo

# Configuration
BENCHMARKS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BENCHMARKS_DIR}/../build"
MODEL_PATH="${BENCHMARKS_DIR}/../models/qwen3-coder-next.gguf"

# Check if build exists
if [ ! -d "$BUILD_DIR" ]; then
    echo -e "${YELLOW}Build directory not found, creating...${NC}"
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    cmake -DGGML_CUDA=ON ..
    make -j$(nproc)
fi

# Find the benchmark executable
BENCHMARK_EXE="${BUILD_DIR}/benchmarks/bench-fattn-gdn"
if [ ! -f "$BENCHMARK_EXE" ]; then
    echo -e "${RED}Benchmark executable not found: $BENCHMARK_EXE${NC}"
    echo "Building benchmark..."
    cd "$BUILD_DIR"
    make bench-fattn-gdn
fi

# Test configurations
declare -a SEQUENCE_LENGTHS=(16 32 64 128 256 512)
declare -a BATCH_SIZES=(1 4 8)

echo -e "${GREEN}Running sequence length benchmarks...${NC}"
echo

# Run sequence length benchmarks
for seq_len in "${SEQUENCE_LENGTHS[@]}"; do
    echo -e "${YELLOW}Sequence length: $seq_len${NC}"
    
    if [ -f "$MODEL_PATH" ]; then
        # Benchmark with actual model (if available)
        echo -e "  Model-based benchmark:"
        "$BENCHMARK_EXE" --model "$MODEL_PATH" --n-prompt "$seq_len" --n-gen 64 --n-batch 512 2>&1 | \
            grep -E "(prompt eval|gen|time|tokens/sec)" || echo "  No benchmark results available"
    fi
    
    # Direct kernel benchmark
    echo -e "  Direct kernel benchmark:"
    "$BENCHMARK_EXE" --kernel-test --n-token "$seq_len" --n-head 32 --s-v 64 2>&1 | \
        grep -E "(kernel|time|tflops)" || echo "  No kernel results available"
    
    echo
done

echo -e "${GREEN}Running batch size benchmarks...${NC}"
echo

# Run batch size benchmarks
for batch_size in "${BATCH_SIZES[@]}"; do
    echo -e "${YELLOW}Batch size: $batch_size${NC}"
    
    if [ -f "$MODEL_PATH" ]; then
        "$BENCHMARK_EXE" --model "$MODEL_PATH" --n-prompt 128 --n-gen 64 --n-batch 512 --n-seq "$batch_size" 2>&1 | \
            grep -E "(batch|time|tokens/sec)" || echo "  No benchmark results available"
    fi
    
    echo
done

echo -e "${GREEN}Running large sequence benchmarks...${NC}"
echo

# Large sequence benchmarks (where flash attention should shine)
LARGE_SEQ_LENGTHS=(1024 2048 4096)
for seq_len in "${LARGE_SEQ_LENGTHS[@]}"; do
    echo -e "${YELLOW}Sequence length: $seq_len${NC}"
    
    if [ -f "$MODEL_PATH" ]; then
        "$BENCHMARK_EXE" --model "$MODEL_PATH" --n-prompt "$seq_len" --n-gen 128 --n-batch 512 2>&1 | \
            grep -E "(prompt eval|gen|time|tokens/sec|flash)" || echo "  No benchmark results available"
    fi
    
    echo
done

echo -e "${GREEN}Running throughput benchmarks...${NC}"
echo

# Throughput benchmark
echo -e "${YELLOW}Throughput test (512 tokens, 10 iterations)${NC}"
if [ -f "$MODEL_PATH" ]; then
    "$BENCHMARK_EXE" --model "$MODEL_PATH" --n-prompt 512 --n-gen 64 --n-batch 512 --n-iter 10 2>&1 | \
        tail -20 || echo "  No throughput results available"
fi

echo
echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}Benchmark Complete${NC}"
echo -e "${BLUE}======================================${NC}"
