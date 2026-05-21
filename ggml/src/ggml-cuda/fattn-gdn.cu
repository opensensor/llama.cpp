#include "fattn-gdn.cuh"

// Flash attention for gated delta net (linear attention layers in Qwen3Next)
// This provides optimized flash attention for the recurrent attention mechanism

template <int DV, bool KDA, bool keep_rs_t, int ncols2>
static void ggml_cuda_fattn_gdn_f32_switch_ncols1(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * Q = dst->src[0];

    // ncols2 is passed as a template parameter (8, 4, 2, or 1)
    // It determines which case threshold to use for Q->ne[1] comparison

    if constexpr (ncols2 <= 8) {
        if (turing_mma_available(cc) && Q->ne[1] <= 8/ncols2) {
            ggml_cuda_fattn_gdn_f32_case<16, DV, 8/ncols2, ncols2>(ctx, dst);
            return;
        }
    }

    if constexpr (ncols2 <= 16) {
        if (Q->ne[1] <= 16/ncols2) {
            ggml_cuda_fattn_gdn_f32_case<32, DV, 16/ncols2, ncols2>(ctx, dst);
            return;
        }
    }

    if (Q->ne[1] <= 32/ncols2 || (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_TURING) ||
            (GGML_CUDA_CC_IS_AMD(cc) && DV > 256)) {
        ggml_cuda_fattn_gdn_f32_case<64, DV, 32/ncols2, ncols2>(ctx, dst);
        return;
    }

    ggml_cuda_fattn_gdn_f32_case<128, DV, 64/ncols2, ncols2>(ctx, dst);
}

template <int DV, bool KDA, bool keep_rs_t>
static void ggml_cuda_fattn_gdn_f32_switch_ncols2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * g    = dst->src[3];
    const ggml_tensor * beta = dst->src[4];
    const ggml_tensor * state = dst->src[5];

    bool use_gqa_opt = K->ne[1] % FATTN_KQ_STRIDE == 0;

    const int gqa_ratio = Q->ne[2] / K->ne[2];

    if (use_gqa_opt && gqa_ratio > 4) {
        ggml_cuda_fattn_gdn_f32_switch_ncols1<DV, KDA, keep_rs_t, 8>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio > 2) {
        ggml_cuda_fattn_gdn_f32_switch_ncols1<DV, KDA, keep_rs_t, 4>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio % 2 == 0) {
        ggml_cuda_fattn_gdn_f32_switch_ncols1<DV, KDA, keep_rs_t, 2>(ctx, dst);
        return;
    }

    ggml_cuda_fattn_gdn_f32_switch_ncols1<DV, KDA, keep_rs_t, 1>(ctx, dst);
}

void ggml_cuda_op_fattn_gdn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float * s_d   = (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    cudaStream_t stream = ctx.stream();

    const int K = (int) src_state->ne[1];
    const bool keep_rs = K > 1;

    if (kda) {
        if (keep_rs) {
            ggml_cuda_fattn_gdn_impl<true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, K, stream);
        } else {
            ggml_cuda_fattn_gdn_impl<true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, K, stream);
        }
    } else {
        if (keep_rs) {
            ggml_cuda_fattn_gdn_impl<false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, K, stream);
        } else {
            ggml_cuda_fattn_gdn_impl<false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, K, stream);
        }
    }
}
