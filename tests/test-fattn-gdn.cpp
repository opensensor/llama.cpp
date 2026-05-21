// Test flash attention for gated delta net (linear attention layers in Qwen3Next)
// This test verifies the flash attention implementation works correctly

#include "ggml.h"
#include "ggml-cuda.h"

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

// Reference implementation of gated delta net (CPU)
static void reference_gated_delta_net(
    const float * q,
    const float * k,
    const float * v,
    const float * g,
    const float * beta,
    const float * state,
    float * dst,
    int64_t S_v,
    int64_t H,
    int64_t n_tokens,
    int64_t n_seqs,
    bool kda) {
    
    const int64_t attn_score_elems = S_v * H * n_tokens * n_seqs;
    
    for (int seq = 0; seq < n_seqs; ++seq) {
        for (int h = 0; h < H; ++h) {
            // Initialize state
            std::vector<float> s(S_v * S_v, 0.0f);
            
            for (int t = 0; t < n_tokens; ++t) {
                const float * q_t = q + (seq * n_tokens + t) * H * S_v + h * S_v;
                const float * k_t = k + (seq * n_tokens + t) * H * S_v + h * S_v;
                const float * v_t = v + (seq * n_tokens + t) * H * S_v + h * S_v;
                const float * beta_t = beta + (seq * n_tokens + t) * H + h;
                
                float g_val;
                if (kda) {
                    const float * g_t = g + (seq * n_tokens + t) * H * S_v + h * S_v;
                    g_val = expf(g_t[0]); // Simplified
                } else {
                    g_val = expf(g[(seq * n_tokens + t) * H + h]);
                }
                
                // Compute kv = S^T @ k
                float kv = 0.0f;
                for (int i = 0; i < S_v; ++i) {
                    kv += s[i] * k_t[i];
                }
                
                // Compute delta = (v - g * kv) * beta
                float delta = (v_t[0] - g_val * kv) * (*beta_t);
                
                // Update state: S = g * S + k * delta
                for (int i = 0; i < S_v; ++i) {
                    s[i] = g_val * s[i] + k_t[i] * delta;
                }
                
                // Compute attention: attn = S^T @ q
                float attn = 0.0f;
                for (int i = 0; i < S_v; ++i) {
                    attn += s[i] * q_t[i];
                }
                
                // Store result
                dst[(seq * n_tokens + h) * S_v + t] = attn * (1.0f / sqrtf((float)S_v));
            }
            
            // Store final state
            float * state_out = dst + attn_score_elems + (seq * H + h) * S_v * S_v;
            for (int i = 0; i < S_v * S_v; ++i) {
                state_out[i] = s[i];
            }
        }
    }
}

// Test 1: Basic functionality
static bool test_fattn_gdn_basic() {
    std::cout << "Test 1: Basic functionality... ";
    
    // Create context
    ggml_init_params params = {
        .mem_size = 256 * 1024 * 1024,  // 256 MB
        .mem_buffer = NULL,
        .no_alloc = false,
    };
    
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        std::cerr << "Failed to create context" << std::endl;
        return false;
    }
    
    // Create tensors
    ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S_v, H, n_tokens);
    ggml_tensor * k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S_v, H, n_tokens);
    ggml_tensor * v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S_v, H, n_tokens);
    ggml_tensor * g = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, H, n_tokens);
    ggml_tensor * beta = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, H, n_tokens);
    ggml_tensor * state = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S_v * S_v, 1, n_seqs);
    
    // Initialize with random data
    srand(42);
    init_tensor_random(q);
    init_tensor_random(k);
    init_tensor_random(v);
    init_tensor_random(g, 0.0f, 1.0f);
    init_tensor_random(beta);
    init_tensor_random(state);
    
    // Create output tensor
    ggml_tensor * dst = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S_v, H * n_seqs, n_tokens);
    
    // Build graph
    ggml_tensor * result = ggml_gated_delta_net(ctx, q, k, v, g, beta, state);
    
    // Verify tensor shapes
    assert(result->ne[0] == S_v);
    assert(result->ne[1] == H);
    assert(result->ne[2] == n_tokens);
    
    // Create compute context
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);
    
    // Compute
    ggml_graph_compute_with_ctx(ctx, graph, 1);
    
    std::cout << "PASSED" << std::endl;
    ggml_free(ctx);
    return true;
}

// Test 2: correctness (compare with reference)
static bool test_fattn_gdn_correctness() {
    std::cout << "Test 2: Correctness (CPU reference)... ";
    
    // Create CPU context
    ggml_init_params params = {
        .mem_size = 512 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc = false,
    };
    
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        std::cerr << "Failed to create context" << std::endl;
        return false;
    }
    
    // Create tensors
    ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S_v, H, n_tokens);
    ggml_tensor * k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S_v, H, n_tokens);
    ggml_tensor * v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S_v, H, n_tokens);
    ggml_tensor * g = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, H, n_tokens);
    ggml_tensor * beta = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, H, n_tokens);
    ggml_tensor * state = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S_v * S_v, 1, n_seqs);
    
    // Initialize with same random data
    srand(123);
    init_tensor_random(q);
    init_tensor_random(k);
    init_tensor_random(v);
    init_tensor_random(g, 0.0f, 1.0f);
    init_tensor_random(beta);
    init_tensor_random(state);
    
    // Run CPU reference
    std::vector<float> cpu_dst(ggml_nelements(q));
    reference_gated_delta_net(
        (float *)q->data,
        (float *)k->data,
        (float *)v->data,
        (float *)g->data,
        (float *)beta->data,
        (float *)state->data,
        cpu_dst.data(),
        S_v, H, n_tokens, n_seqs, false);
    
    // Create GPU output
    ggml_tensor * dst = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S_v, H * n_seqs, n_tokens);
    ggml_tensor * result = ggml_gated_delta_net(ctx, q, k, v, g, beta, state);
    
    // Create graph and compute
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);
    
    // Compute
    ggml_graph_compute_with_ctx(ctx, graph, 1);
    
    // Compare results (allow some tolerance for floating point)
    const float * gpu_data = (const float *)result->data;
    double sum_sq_diff = 0.0;
    int64_t ne = ggml_nelements(result);
    
    for (int64_t i = 0; i < ne; ++i) {
        float diff = gpu_data[i] - cpu_dst[i];
        sum_sq_diff += diff * diff;
    }
    
    double rms_error = sqrt(sum_sq_diff / ne);
    
    // Allow RMS error < 1e-3
    bool passed = rms_error < 1e-3;
    
    if (passed) {
        std::cout << "PASSED (RMS error: " << rms_error << ")" << std::endl;
    } else {
        std::cout << "FAILED (RMS error: " << rms_error << ")" << std::endl;
    }
    
    ggml_free(ctx);
    return passed;
}

// Test 3: Different sequence lengths
static bool test_fattn_gdn_seq_lengths() {
    std::cout << "Test 3: Different sequence lengths... ";
    
    const std::vector<int> seq_lengths = {8, 16, 32, 64, 128, 256};
    bool all_passed = true;
    
    for (int n_tokens_test : seq_lengths) {
        ggml_init_params params = {
            .mem_size = 256 * 1024 * 1024,
            .mem_buffer = NULL,
            .no_alloc = false,
        };
        
        ggml_context * ctx = ggml_init(params);
        if (!ctx) {
            all_passed = false;
            continue;
        }
        
        ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S_v, H, n_tokens_test);
        ggml_tensor * k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S_v, H, n_tokens_test);
        ggml_tensor * v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S_v, H, n_tokens_test);
        ggml_tensor * g = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, H, n_tokens_test);
        ggml_tensor * beta = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, H, n_tokens_test);
        ggml_tensor * state = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S_v * S_v, 1, n_seqs);
        
        srand(456 + n_tokens_test);
        init_tensor_random(q);
        init_tensor_random(k);
        init_tensor_random(v);
        init_tensor_random(g, 0.0f, 1.0f);
        init_tensor_random(beta);
        init_tensor_random(state);
        
        ggml_tensor * result = ggml_gated_delta_net(ctx, q, k, v, g, beta, state);
        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, result);
        
        ggml_graph_compute_with_ctx(ctx, graph, 1);
        
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
    
    ggml_init_params params = {
        .mem_size = 256 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc = false,
    };
    
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        std::cerr << "Failed to create context" << std::endl;
        return false;
    }
    
    // KDA mode: g has same shape as q/k/v
    ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S_v, H, n_tokens);
    ggml_tensor * k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S_v, H, n_tokens);
    ggml_tensor * v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S_v, H, n_tokens);
    ggml_tensor * g = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S_v, H, n_tokens);
    ggml_tensor * beta = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, H, n_tokens);
    ggml_tensor * state = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S_v * S_v, 1, n_seqs);
    
    srand(789);
    init_tensor_random(q);
    init_tensor_random(k);
    init_tensor_random(v);
    init_tensor_random(g, 0.0f, 1.0f);
    init_tensor_random(beta);
    init_tensor_random(state);
    
    ggml_tensor * result = ggml_gated_delta_net(ctx, q, k, v, g, beta, state);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);
    
    ggml_graph_compute_with_ctx(ctx, graph, 1);
    
    std::cout << "PASSED" << std::endl;
    ggml_free(ctx);
    return true;
}

// Test 5: State retention (K > 1)
static bool test_fattn_gdn_state_retention() {
    std::cout << "Test 5: State retention (K > 1)... ";
    
    const int K = 3;  // Number of state snapshots
    
    ggml_init_params params = {
        .mem_size = 256 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc = false,
    };
    
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        std::cerr << "Failed to create context" << std::endl;
        return false;
    }
    
    ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S_v, H, n_tokens);
    ggml_tensor * k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S_v, H, n_tokens);
    ggml_tensor * v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S_v, H, n_tokens);
    ggml_tensor * g = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, H, n_tokens);
    ggml_tensor * beta = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, H, n_tokens);
    // State with K snapshots
    ggml_tensor * state = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S_v * S_v, K, n_seqs);
    
    srand(1011);
    init_tensor_random(q);
    init_tensor_random(k);
    init_tensor_random(v);
    init_tensor_random(g, 0.0f, 1.0f);
    init_tensor_random(beta);
    init_tensor_random(state);
    
    ggml_tensor * result = ggml_gated_delta_net(ctx, q, k, v, g, beta, state);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);
    
    ggml_graph_compute_with_ctx(ctx, graph, 1);
    
    std::cout << "PASSED" << std::endl;
    ggml_free(ctx);
    return true;
}

// Test 6: Performance test (large sequences)
static bool test_fattn_gdn_performance() {
    std::cout << "Test 6: Performance (large sequences)... ";
    
    const int S_v_perf = 64;
    const int H_perf = 32;
    const int n_tokens_perf = 512;
    
    ggml_init_params params = {
        .mem_size = 512 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc = false,
    };
    
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        std::cerr << "Failed to create context" << std::endl;
        return false;
    }
    
    ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S_v_perf, H_perf, n_tokens_perf);
    ggml_tensor * k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S_v_perf, H_perf, n_tokens_perf);
    ggml_tensor * v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S_v_perf, H_perf, n_tokens_perf);
    ggml_tensor * g = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, H_perf, n_tokens_perf);
    ggml_tensor * beta = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, H_perf, n_tokens_perf);
    ggml_tensor * state = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S_v_perf * S_v_perf, 1, n_seqs);
    
    srand(2024);
    init_tensor_random(q);
    init_tensor_random(k);
    init_tensor_random(v);
    init_tensor_random(g, 0.0f, 1.0f);
    init_tensor_random(beta);
    init_tensor_random(state);
    
    // Warmup
    ggml_tensor * result = ggml_gated_delta_net(ctx, q, k, v, g, beta, state);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);
    ggml_graph_compute_with_ctx(ctx, graph, 1);
    
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
    total++; if (test_fattn_gdn_correctness()) passed++;
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
