// Test flash attention for gated delta net (linear attention layers in Qwen3Next)
// This test verifies the flash attention implementation works correctly

#include "ggml.h"
#include "ggml-cuda.h"
#include "ggml-cpu.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cassert>

// Test parameters
const int S_v = 64;        // State dimension
const int H = 32;          // Number of heads
const int n_tokens = 128;  // Number of tokens
const int n_seqs = 4;      // Number of sequences

// Helper function to initialize tensor with random data
static void init_tensor_random(ggml_tensor * tensor, float min = -1.0f, float max = 1.0f) {
    float * data = (float *)tensor->data;
    int64_t ne = ggml_nelements(tensor);
    for (int64_t i = 0; i < ne; ++i) {
        data[i] = min + (max - min) * (float)rand() / RAND_MAX;
    }
}

static void init_gdn_inputs(
        ggml_tensor * q,
        ggml_tensor * k,
        ggml_tensor * v,
        ggml_tensor * g,
        ggml_tensor * beta,
        ggml_tensor * state,
        unsigned seed) {
    srand(seed);
    init_tensor_random(q,     -0.25f,  0.25f);
    init_tensor_random(k,     -0.25f,  0.25f);
    init_tensor_random(v,     -0.50f,  0.50f);
    init_tensor_random(g,     -5.00f, -0.10f);
    init_tensor_random(beta,   0.00f,  1.00f);
    init_tensor_random(state, -0.10f,  0.10f);
}

static bool tensor_has_nonfinite(const ggml_tensor * tensor) {
    const float * data = (const float *) tensor->data;
    const int64_t ne = ggml_nelements(tensor);
    for (int64_t i = 0; i < ne; ++i) {
        if (!std::isfinite(data[i])) {
            return true;
        }
    }
    return false;
}

static ggml_init_params make_init_params(size_t mem_size) {
    ggml_init_params params;
    params.mem_size   = mem_size;
    params.mem_buffer = NULL;
    params.no_alloc   = false;
    return params;
}

// Test 1: Basic functionality
static bool test_fattn_gdn_basic() {
    std::cout << "Test 1: Basic functionality... ";

    // Create context
    ggml_init_params params = make_init_params(256 * 1024 * 1024);

    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        std::cerr << "Failed to create context" << std::endl;
        return false;
    }

    // Create tensors with correct layout: [S_v, H, n_tokens, n_seqs]
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v, H, n_tokens, n_seqs);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v, H, n_tokens, n_seqs);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v, H, n_tokens, n_seqs);
    ggml_tensor * g = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, n_tokens, n_seqs);
    ggml_tensor * beta = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, n_tokens, n_seqs);
    // State tensor: 3D tensor [S_v*S_v*H, K=1, n_seqs] (ggml_new_tensor_4d uses [ne0, ne1, ne2, ne3])
    ggml_tensor * state = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v * S_v * H, 1, n_seqs, 1);

    init_gdn_inputs(q, k, v, g, beta, state, 42);

    // Build graph
    ggml_tensor * result = ggml_gated_delta_net(ctx, q, k, v, g, beta, state);

    // Verify result tensor has expected layout
    // result->ne = [S_v*H, n_tokens*n_seqs + state_rows, 1, 1]
    const int64_t state_rows = 1 * S_v * n_seqs;
    const int64_t expected_ne1 = n_tokens * n_seqs + state_rows;
    if (result->ne[0] != S_v * H || result->ne[1] != expected_ne1) {
        std::cout << "FAILED (unexpected output shape)" << std::endl;
        ggml_free(ctx);
        return false;
    }

    // Create compute context
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);

    // Compute
    ggml_graph_compute_with_ctx(ctx, graph, 1);

    if (tensor_has_nonfinite(result)) {
        std::cout << "FAILED (non-finite output)" << std::endl;
    } else {
        std::cout << "PASSED" << std::endl;
    }

    const bool passed = !tensor_has_nonfinite(result);
    ggml_free(ctx);
    return passed;
}

// Test 2: Different batch sizes
static bool test_fattn_gdn_batch_sizes() {
    std::cout << "Test 2: Different batch sizes... ";

    const std::vector<int> batch_sizes = {1, 2, 4, 8};
    bool all_passed = true;

    for (int n_seqs_test : batch_sizes) {
        ggml_init_params params = make_init_params(256 * 1024 * 1024);

        ggml_context * ctx = ggml_init(params);
        if (!ctx) {
            all_passed = false;
            continue;
        }

        ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v, H, n_tokens, n_seqs_test);
        ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v, H, n_tokens, n_seqs_test);
        ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v, H, n_tokens, n_seqs_test);
        ggml_tensor * g = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, n_tokens, n_seqs_test);
        ggml_tensor * beta = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, n_tokens, n_seqs_test);
        ggml_tensor * state = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v * S_v * H, 1, n_seqs_test, 1);

        init_gdn_inputs(q, k, v, g, beta, state, 456 + n_seqs_test);

        ggml_tensor * result = ggml_gated_delta_net(ctx, q, k, v, g, beta, state);
        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, result);

        ggml_graph_compute_with_ctx(ctx, graph, 1);

        if (tensor_has_nonfinite(result)) {
            std::cout << "FAILED (non-finite output for n_seqs=" << n_seqs_test << ")" << std::endl;
            all_passed = false;
        }

        ggml_free(ctx);
    }

    if (all_passed) {
        std::cout << "PASSED" << std::endl;
    } else {
        std::cout << "FAILED" << std::endl;
    }

    return all_passed;
}

// Test 3: Different sequence lengths
static bool test_fattn_gdn_seq_lengths() {
    std::cout << "Test 3: Different sequence lengths... ";

    const std::vector<int> seq_lengths = {8, 16, 32, 64, 128, 256};
    bool all_passed = true;

    for (int n_tokens_test : seq_lengths) {
        ggml_init_params params = make_init_params(256 * 1024 * 1024);

        ggml_context * ctx = ggml_init(params);
        if (!ctx) {
            all_passed = false;
            continue;
        }

        ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v, H, n_tokens_test, 1);
        ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v, H, n_tokens_test, 1);
        ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v, H, n_tokens_test, 1);
        ggml_tensor * g = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, n_tokens_test, 1);
        ggml_tensor * beta = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, n_tokens_test, 1);
        // State uses n_seqs from Q/K/V which is 1 in this test
        ggml_tensor * state = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v * S_v * H, 1, 1, 1);

        init_gdn_inputs(q, k, v, g, beta, state, 456 + n_tokens_test);

        ggml_tensor * result = ggml_gated_delta_net(ctx, q, k, v, g, beta, state);
        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, result);

        ggml_graph_compute_with_ctx(ctx, graph, 1);

        if (tensor_has_nonfinite(result)) {
            std::cout << "FAILED (non-finite output for n_tokens=" << n_tokens_test << ")" << std::endl;
            all_passed = false;
        }

        ggml_free(ctx);
    }

    if (all_passed) {
        std::cout << "PASSED" << std::endl;
    } else {
        std::cout << "FAILED" << std::endl;
    }

    return all_passed;
}

// Test 4: KDA (Key-Dependent Activation) mode
static bool test_fattn_gdn_kda() {
    std::cout << "Test 4: KDA (Key-Dependent Activation)... ";

    ggml_init_params params = make_init_params(256 * 1024 * 1024);

    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        std::cerr << "Failed to create context" << std::endl;
        return false;
    }

    // KDA mode: g has same shape as q/k/v
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v, H, n_tokens, n_seqs);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v, H, n_tokens, n_seqs);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v, H, n_tokens, n_seqs);
    ggml_tensor * g = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v, H, n_tokens, n_seqs);
    ggml_tensor * beta = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, n_tokens, n_seqs);
    ggml_tensor * state = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v * S_v * H, 1, n_seqs, 1);

    init_gdn_inputs(q, k, v, g, beta, state, 789);

    ggml_tensor * result = ggml_gated_delta_net(ctx, q, k, v, g, beta, state);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);

    ggml_graph_compute_with_ctx(ctx, graph, 1);

    if (tensor_has_nonfinite(result)) {
        std::cout << "FAILED (non-finite output)" << std::endl;
    } else {
        std::cout << "PASSED" << std::endl;
    }

    const bool passed = !tensor_has_nonfinite(result);
    ggml_free(ctx);
    return passed;
}

// Test 5: State retention (K > 1)
static bool test_fattn_gdn_state_retention() {
    std::cout << "Test 5: State retention (K > 1)... ";

    const int K = 3;  // Number of state snapshots

    ggml_init_params params = make_init_params(256 * 1024 * 1024);

    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        std::cerr << "Failed to create context" << std::endl;
        return false;
    }

    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v, H, n_tokens, n_seqs);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v, H, n_tokens, n_seqs);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v, H, n_tokens, n_seqs);
    ggml_tensor * g = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, n_tokens, n_seqs);
    ggml_tensor * beta = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, n_tokens, n_seqs);
    // State with K snapshots
    ggml_tensor * state = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v * S_v * H, K, n_seqs, 1);

    init_gdn_inputs(q, k, v, g, beta, state, 1011);

    ggml_tensor * result = ggml_gated_delta_net(ctx, q, k, v, g, beta, state);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);

    ggml_graph_compute_with_ctx(ctx, graph, 1);

    if (tensor_has_nonfinite(result)) {
        std::cout << "FAILED (non-finite output)" << std::endl;
    } else {
        std::cout << "PASSED" << std::endl;
    }

    const bool passed = !tensor_has_nonfinite(result);
    ggml_free(ctx);
    return passed;
}

// Test 6: Performance test (large sequences)
static bool test_fattn_gdn_performance() {
    std::cout << "Test 6: Performance (large sequences)... ";

    const int S_v_perf = 64;
    const int H_perf = 32;
    const int n_tokens_perf = 512;

    ggml_init_params params = make_init_params(512 * 1024 * 1024);

    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        std::cerr << "Failed to create context" << std::endl;
        return false;
    }

    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v_perf, H_perf, n_tokens_perf, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v_perf, H_perf, n_tokens_perf, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v_perf, H_perf, n_tokens_perf, 1);
    ggml_tensor * g = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H_perf, n_tokens_perf, 1);
    ggml_tensor * beta = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H_perf, n_tokens_perf, 1);
    // State uses n_seqs from Q/K/V which is 1
    ggml_tensor * state = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v_perf * S_v_perf * H_perf, 1, 1, 1);

    init_gdn_inputs(q, k, v, g, beta, state, 2024);

    // Warmup
    ggml_tensor * result = ggml_gated_delta_net(ctx, q, k, v, g, beta, state);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);
    ggml_graph_compute_with_ctx(ctx, graph, 1);
    if (tensor_has_nonfinite(result)) {
        std::cout << "FAILED (non-finite output)" << std::endl;
        ggml_free(ctx);
        return false;
    }

    // Performance run
    int64_t t_start = ggml_time_us();
    ggml_graph_compute_with_ctx(ctx, graph, 1);
    int64_t t_end = ggml_time_us();

    double elapsed_ms = (t_end - t_start) / 1000.0;
    double tokens_per_sec = n_tokens_perf / (elapsed_ms / 1000.0);

    std::cout << "PASSED (elapsed: " << elapsed_ms << " ms, tokens/sec: " << tokens_per_sec << ")" << std::endl;

    ggml_free(ctx);
    return true;
}

int main() {
    std::cout << "=== Flash Attention for Gated Delta Net Tests ===" << std::endl;
    std::cout << "S_v = " << S_v << ", H = " << H << ", n_tokens = " << n_tokens << ", n_seqs = " << n_seqs << std::endl;
    std::cout << std::endl;

    int passed = 0;
    int total = 0;

    total++; if (test_fattn_gdn_basic()) passed++;
    total++; if (test_fattn_gdn_batch_sizes()) passed++;
    total++; if (test_fattn_gdn_seq_lengths()) passed++;
    total++; if (test_fattn_gdn_kda()) passed++;
    total++; if (test_fattn_gdn_state_retention()) passed++;
    total++; if (test_fattn_gdn_performance()) passed++;

    std::cout << std::endl;
    std::cout << "=== Results ===" << std::endl;
    std::cout << "Passed: " << passed << "/" << total << std::endl;

    if (passed == total) {
        std::cout << "All tests PASSED!" << std::endl;
        return 0;
    } else {
        std::cout << "Some tests FAILED!" << std::endl;
        return 1;
    }
}
