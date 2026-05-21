#pragma once

#include "common.cuh"
#include "fattn-common.cuh"

// Flash attention for gated delta net (linear attention)
// Combines SSM convolution with flash attention for improved performance

template <int S_v, int ncols2>
static __device__ void ggml_cuda_fattn_gdn_qk_f32(
    const float * q,
    const float * k,
    float *       kq,
    int64_t       stride_k,
    int64_t       stride_kq,
    int           n_tokens,
    int           n_heads,
    int           n_seqs,
    int           head_idx,
    int           seq_idx,
    int           col_start,
    int           row) {
    const int col = col_start + threadIdx.y;

    float kq_val = 0.0f;
#pragma unroll
    for (int i = 0; i < S_v; ++i) {
        const float q_val = q[row * S_v + i];
        const float k_val = k[col * stride_k + i];
        kq_val += q_val * k_val;
    }

    kq[row * stride_kq + col] = kq_val;
}

template <int S_v, int ncols2>
static __device__ void ggml_cuda_fattn_gdn_attn_f32(
    const float * kq,
    const float * v,
    float *       dst,
    int64_t       stride_v,
    int64_t       stride_dst,
    int           n_tokens,
    int           n_heads,
    int           n_seqs,
    int           head_idx,
    int           seq_idx,
    int           row,
    float         scale) {
    const int col = threadIdx.y;

    float dst_val = 0.0f;
#pragma unroll
    for (int i = 0; i < n_tokens; ++i) {
        const float kq_val = kq[row * n_tokens + i];
        const float v_val = v[col * stride_v + i];
        dst_val += kq_val * v_val;
    }

    dst[col * stride_dst + row] = dst_val * scale;
}

template <int S_v, bool KDA, bool keep_rs_t>
static __global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
ggml_cuda_fattn_gdn_f32(
    const float * q,
    const float * k,
    const float * v,
    const float * g,
    const float * beta,
    const float * curr_state,
    float *       dst,
    int64_t       H,
    int64_t       n_tokens,
    int64_t       n_seqs,
    int64_t       sq1,
    int64_t       sq2,
    int64_t       sq3,
    int64_t       sv1,
    int64_t       sv2,
    int64_t       sv3,
    int64_t       sb1,
    int64_t       sb2,
    int64_t       sb3,
    const uint3   neqk1_magic,
    const uint3   rq3_magic,
    int           K) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    const int      lane     = threadIdx.x;
    const int      col      = blockIdx.z * blockDim.y + threadIdx.y;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    const int64_t attn_score_elems = S_v * H * n_tokens * n_seqs;
    float *       attn_data        = dst;
    float *       state            = dst + attn_score_elems;

    const int64_t state_in_offset      = sequence * K * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    const int64_t state_size_per_token = S_v * S_v * H * n_seqs;
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    float         s_shard[rows_per_lane];

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        s_shard[r]  = curr_state[i];
    }

    const int shift = (int) n_tokens - K;

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        const float beta_val = *beta_t;

        // Cache k and q in registers
        float q_reg[rows_per_lane];
        float k_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            q_reg[r] = q_t[i];
            k_reg[r] = k_t[i];
        }

        if constexpr (!KDA) {
            const float g_val = expf(*g_t);

            // kv = S^T @ k
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += s_shard[r] * k_reg[r];
            }
            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta = (v - g * kv) * beta
            float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            // S = g * S + k * delta
            // attn = S^T @ q
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * (1.0f / sqrtf((float) S_v));
            }
        } else {
            // kv = sum_i g[i] * S[i] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += expf(g_t[r * warp_size + lane]) * s_shard[r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta = (v - kv) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // S = g * S + k * delta
            // attn = S^T @ q
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]  = expf(g_t[r * warp_size + lane]) * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * (1.0f / sqrtf((float) S_v));
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            const int target_slot = t - shift;
            if (target_slot >= 0 && target_slot < K) {
                float * curr_state = (dst + attn_score_elems) + target_slot * state_size_per_token + state_out_offset;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * warp_size + lane;
                    curr_state[col * S_v + i] = s_shard[r];
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            state[col * S_v + i] = s_shard[r];
        }
    }
}

template <bool KDA, bool keep_rs_t>
static void ggml_cuda_fattn_gdn_impl(
    const float * q_d, const float * k_d, const float * v_d,
    const float * g_d, const float * b_d, const float * s_d,
    float * dst_d,
    int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
    int64_t sq1,   int64_t sq2, int64_t sq3,
    int64_t sv1,   int64_t sv2, int64_t sv3,
    int64_t sb1,   int64_t sb2, int64_t sb3,
    int64_t neqk1, int64_t rq3,
    int K, cudaStream_t stream) {
    const int num_warps = 4;
    dim3      grid_dims(H, n_seqs, (S_v + num_warps - 1) / num_warps);
    dim3      block_dims(32 <= S_v ? 32 : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);

    switch (S_v) {
        case 16:
            ggml_cuda_kernel_launch(ggml_cuda_fattn_gdn_f32<16, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, K);
            break;
        case 32:
            ggml_cuda_kernel_launch(ggml_cuda_fattn_gdn_f32<32, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, K);
            break;
        case 64:
            ggml_cuda_kernel_launch(ggml_cuda_fattn_gdn_f32<64, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, K);
            break;
        case 128:
            ggml_cuda_kernel_launch(ggml_cuda_fattn_gdn_f32<128, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, K);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

// Forward declarations for switch functions (defined in fattn-gdn.cu)
void ggml_cuda_op_fattn_gdn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

template <int DV, bool KDA, bool keep_rs_t>
void ggml_cuda_fattn_gdn_f32_switch_ncols2(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

template <int DV, bool KDA, bool keep_rs_t, int ncols2>
void ggml_cuda_fattn_gdn_f32_switch_ncols1(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

template <int DKQ, int DV, int ncols2>
void ggml_cuda_fattn_gdn_f32_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    const bool kda = (dst->src[3]->ne[0] == DV);

    if (kda) {
        if (ncols2 == 1) {
            ggml_cuda_fattn_gdn_f32_switch_ncols2<DV, true, true>(ctx, dst);
        } else {
            ggml_cuda_fattn_gdn_f32_switch_ncols2<DV, true, false>(ctx, dst);
        }
    } else {
        if (ncols2 == 1) {
            ggml_cuda_fattn_gdn_f32_switch_ncols2<DV, false, true>(ctx, dst);
        } else {
            ggml_cuda_fattn_gdn_f32_switch_ncols2<DV, false, false>(ctx, dst);
        }
    }
}

#define DECL_FATTN_GDN_CASE(DKQ, DV, ncols2)                              \
    template void ggml_cuda_fattn_gdn_f32_case                            \
    <DKQ, DV, ncols2>(ggml_backend_cuda_context & ctx, ggml_tensor * dst) \

extern DECL_FATTN_GDN_CASE( 16,  16, 8);
extern DECL_FATTN_GDN_CASE( 32,  32, 16);
extern DECL_FATTN_GDN_CASE( 64,  64, 32);
extern DECL_FATTN_GDN_CASE(128, 128, 64);
