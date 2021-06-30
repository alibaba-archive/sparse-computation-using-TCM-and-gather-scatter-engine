

/*
 * Copyright (c) 2020
 * All rights reserved
 *
 * The license below extends only to copyright in the software and shall
 * not be construed as granting a license to any other intellectual
 * property including but not limited to intellectual property relating
 * to a hardware implementation of the functionality of the software
 * licensed hereunder.  You may use the software subject to the license
 * terms below provided that you ensure that this notice is replicated
 * unmodified and in its entirety in all distributions of the software,
 * modified or unmodified, in source code or in binary form.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met: redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer;
 * redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution;
 * neither the name of the copyright holders nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */
#include <cassert>
#include <iostream>
#include <utility>
#include "mm.h"

#define TRY_SPLIT 0

#if COMPUTE_FP32
inline void sp_row_joined(const float16_t *A,
    const unsigned short *B_join,
    unsigned int start, unsigned int end, unsigned int nitems,
    float16_t *Out_addr, bool reduction) {

//   const svbool_t pred = svptrue_b32();
  svfloat32_t acc = svdup_f32(0);
  svfloat32_t acc1 = svdup_f32(0);
  const svbool_t pred16 = svptrue_b16();
  const svbool_t pred32 = svptrue_b32();
  int step = 2 * nitems;
  const unsigned short * from = &B_join[start * step];
  const unsigned short * to = &B_join[end * step];
  for (const unsigned short *p = from; p != to; p+= step) {
    svuint16_t B_k = svld1_vnum_u16(pred16, p, 1);
    svuint16_t B_v_u = svld1_vnum_u16(pred16, p, 0);
       //load weight value, 0 means no displacement from the base
    svfloat16_t B_v = svreinterpret_f16(B_v_u);
    svfloat16_t B_v_odd16 = svtrn2(B_v, B_v);
    const svfloat32_t B_v_even = svcvt_f32_x(pred16, B_v);
    const svfloat32_t B_v_odd = svcvt_f32_x(pred16, B_v_odd16);

       //load weight index, 1 means 1 vnum displacement from the base,
       // as we joinly store the indices and values of the sparse matrix
    svfloat16_t A_g = svreinterpret_f16(
        svld1_gather_index(pred16, (float64_t *)A, svreinterpret_u64(B_k)));
    svfloat16_t A_g_odd16 = svtrn2(A_g, A_g);
    const svfloat32_t A_g_even = svcvt_f32_x(pred16, A_g);
    const svfloat32_t A_g_odd = svcvt_f32_x(pred16, A_g_odd16);

        //gather from dense activation vector
    acc = svmla_x(pred32, acc, A_g_even, B_v_even); //perform mla
    acc1 = svmla_x(pred32, acc1, A_g_odd, B_v_odd); //perform mla
  }

  svfloat16_t acc16_odd = svcvt_f16_z(pred16, acc);
  svfloat16_t acc16_even = svcvt_f16_z(pred16, acc1);
  if (reduction) {
    svfloat16_t acc16 = svadd_z(pred32, acc16_even, acc16_odd);
    float32_t r = svaddv(pred32, acc16);
    *Out_addr = r;
  } else {
    svfloat16_t acc16 = svtrn1(acc16_odd, acc16_even);
    // svfloat16_t c = svld1_f16(pred16, Out_addr);
    // acc16 = svadd_f16_z(pred16, acc16, c);
    svst1(pred16, Out_addr, acc16);
  }
}
inline void sp_row_joined_fused(const float16_t *A,
    const unsigned short *B_join,
    unsigned int start, unsigned int end, unsigned int nitems,
    float16_t *Out_addr, bool reduction)
{
    std::cout<<"sp_row_joined_fused for FP32 NOT IMPLEMENTED"<<std::endl;
}

inline void sp_row_joined_scatter(const float16_t *A,
    const unsigned short *B_join,
    const unsigned short *B_rowind,
    unsigned int start_rowind,
    unsigned int start, unsigned int end, unsigned int nitems,
    float16_t *Out_addr, bool reduction) {

   const svbool_t pred = svptrue_b32();
  svfloat32_t acc = svdup_f32(0);
  svfloat32_t acc1 = svdup_f32(0);
  const svbool_t pred16lo = svdupq_b16(false, false, false, false,
                                       true, true, true, true);
  const svbool_t pred16hi = svdupq_b16(true, true, true, true,
                                       false, false, false, false);
  int step = 2 * nitems;
  const unsigned short * from = &B_join[start * step];
  const unsigned short * to = &B_join[end * step];
  svuint16_t B_ri = svld1_u16(pred, &B_rowind[start_rowind]);
  for (const unsigned short *p = from; p != to; p+= step) {
    svuint16_t B_k = svld1_vnum_u16(pred, p, 1);
    svuint16_t B_v_u = svld1_vnum_u16(pred, p, 0);
       //load weight value, 0 means no displacement from the base
    svfloat16_t B_v = svreinterpret_f16(B_v_u);
    const svfloat32_t B_v_hi = svcvt_f32_x(pred16hi, B_v);
    const svfloat32_t B_v_lo = svcvt_f32_x(pred16lo, B_v);

       //load weight index, 1 means 1 vnum displacement from the base,
       // as we joinly store the indices and values of the sparse matrix
    svfloat16_t A_g = svreinterpret_f16(
        svld1_gather_index(pred, (float64_t *)A, svreinterpret_u64(B_k)));
    const svfloat32_t A_g_hi = svcvt_f32_x(pred16hi, A_g);
    const svfloat32_t A_g_lo = svcvt_f32_x(pred16lo, A_g);
        //gather from dense activation vector
    acc = svmla_x(pred, acc, A_g_hi, B_v_hi); //perform mla
    acc1 = svmla_x(pred, acc1, A_g_lo, B_v_lo); //perform mla
  }

  if (reduction) {
    acc = svadd_z(pred, acc, acc1);
    float32_t r = svaddv(pred, acc);
    *Out_addr = r;
  } else {
    svfloat16_t acc16 = svcvt_f16_z(pred16lo, acc1);
    acc16 = svcvt_f16_m(acc16, pred16hi, acc);
    // svst1(pred, Out_addr, acc16);
    svst1_scatter_index(pred, (float64_t *)Out_addr,
      svreinterpret_u64(B_ri), svreinterpret_f64(acc16));
  }
}

#else

inline void sp_row_joined_scatter(const float16_t *A,
    const unsigned short *B_join,
    const unsigned short *B_rowind,
    unsigned int start_rowind,
    unsigned int start, unsigned int end, unsigned int nitems,
    float16_t *Out_addr, bool reduction) {

  svfloat16_t acc = svdup_f16(0);
  const svbool_t pred = svptrue_b16();

  int step = 2 * nitems;
  const unsigned short * from = &B_join[start * step];
  const unsigned short * to = &B_join[end * step];

  svuint16_t B_ri = svld1_u16(pred, &B_rowind[start_rowind]);
#if TRY_SPLIT
  svfloat16_t acc1 = svdup_f16(0);
  const unsigned short *p = from;
  if (((end - start) % 2) == 1) {
    svuint16_t B_v_u = svld1_vnum_u16(pred, p, 0);
       //load weight value, 0 means no displacement from the base
    svfloat16_t B_v = svreinterpret_f16(B_v_u);
    svuint16_t B_k = svld1_vnum_u16(pred, p, 1);
       //load weight index, 1 means 1 vnum displacement from the base,
       // as we joinly store the indices and values of the sparse matrix
    svfloat16_t A_g = svreinterpret_f16(
        svld1_gather_index(pred, (float64_t *)A, svreinterpret_u64(B_k)));
        //gather from dense activation vector
    acc = svmla_x(pred, acc, A_g, B_v); //perform mla
    p += step;
  }
  for (; p != to; p+= 2 * step) {
    svuint16_t B_v_u_0 = svld1_vnum_u16(pred, p, 0);
       //load weight value, 0 means no displacement from the base
    svfloat16_t B_v_0 = svreinterpret_f16(B_v_u_0);
    svuint16_t B_k_0 = svld1_vnum_u16(pred, p, 1);

    svuint16_t B_v_u_1 = svld1_vnum_u16(pred, p, 2);
       //load weight value, 0 means no displacement from the base
    svfloat16_t B_v_1 = svreinterpret_f16(B_v_u_1);
    svuint16_t B_k_1 = svld1_vnum_u16(pred, p, 3);

       //load weight index, 1 means 1 vnum displacement from the base,
       // as we joinly store the indices and values of the sparse matrix
    svfloat16_t A_g_0 = svreinterpret_f16(
        svld1_gather_index(pred, (float64_t *)A, svreinterpret_u64(B_k_0)));
    svfloat16_t A_g_1 = svreinterpret_f16(
        svld1_gather_index(pred, (float64_t *)A, svreinterpret_u64(B_k_1)));
        //gather from dense activation vector
    acc = svmla_x(pred, acc, A_g_0, B_v_0); //perform mla
    acc1 = svmla_x(pred, acc1, A_g_1, B_v_1); //perform mla
  }
  acc = svadd_z(pred, acc, acc1);

#else
  for (const unsigned short *p = from; p != to; p+= step) {
    /*
    const float16_t *pp = (const float16_t *)p;
    std::cout << pp[0] << ":" << pp[1] << ":" << pp[2] << ":" << pp[3] << ":"
    << pp[4] << ":" << pp[5] << ":" << pp[6] << ":" << pp[7] << std::endl;
    const unsigned short *ppp = &p[8];
    std::cout << ppp[0] << ":" << ppp[1] << ":" << ppp[2] << ":"
    << ppp[3] << ":"
    << ppp[4] << ":" << ppp[5] << ":" << ppp[6] << ":" << ppp[7] << std::endl;
    */
    svuint16_t B_v_u = svld1_vnum_u16(pred, p, 0);
       //load weight value, 0 means no displacement from the base
    svfloat16_t B_v = svreinterpret_f16(B_v_u);
    svuint16_t B_k = svld1_vnum_u16(pred, p, 1);
       //load weight index, 1 means 1 vnum displacement from the base,
       // as we joinly store the indices and values of the sparse matrix
    svfloat16_t A_g = svreinterpret_f16(
        svld1_gather_index(pred, (float64_t *)A, svreinterpret_u64(B_k)));
        //gather from dense activation vector
    acc = svmla_x(pred, acc, A_g, B_v); //perform mla

    // svst1(true32, &C[i*vitems], A_g);
  }
#endif

  if (reduction) {
    float16_t r = svaddv(pred, acc);
    *Out_addr = r;
  } else {
    // svst1(pred, Out_addr, acc);
    svst1_scatter_index(pred, (float64_t *)Out_addr,
          svreinterpret_u64(B_ri), svreinterpret_f64(acc));
  }
}

inline void sp_row_joined(const float16_t *A,
    const unsigned short *B_join,
    unsigned int start, unsigned int end, unsigned int nitems,
    float16_t *Out_addr, bool reduction) {

  svfloat16_t acc = svdup_f16(0);
  const svbool_t pred = svptrue_b16();

  int step = 2 * nitems;
  const unsigned short * from = &B_join[start * step];
  const unsigned short * to = &B_join[end * step];
#if TRY_SPLIT
  svfloat16_t acc1 = svdup_f16(0);
  const unsigned short *p = from;
  if (((end - start) % 2) == 1) {
    svuint16_t B_v_u = svld1_vnum_u16(pred, p, 0);
       //load weight value, 0 means no displacement from the base
    svfloat16_t B_v = svreinterpret_f16(B_v_u);
    svuint16_t B_k = svld1_vnum_u16(pred, p, 1);
       //load weight index, 1 means 1 vnum displacement from the base,
       // as we joinly store the indices and values of the sparse matrix
    svfloat16_t A_g = svreinterpret_f16(
        svld1_gather_index(pred, (float64_t *)A, svreinterpret_u64(B_k)));
        //gather from dense activation vector
    acc = svmla_x(pred, acc, A_g, B_v); //perform mla
    p += step;
  }
  for (; p != to; p+= 2 * step) {
    svuint16_t B_v_u_0 = svld1_vnum_u16(pred, p, 0);
       //load weight value, 0 means no displacement from the base
    svfloat16_t B_v_0 = svreinterpret_f16(B_v_u_0);
    svuint16_t B_k_0 = svld1_vnum_u16(pred, p, 1);

    svuint16_t B_v_u_1 = svld1_vnum_u16(pred, p, 2);
       //load weight value, 0 means no displacement from the base
    svfloat16_t B_v_1 = svreinterpret_f16(B_v_u_1);
    svuint16_t B_k_1 = svld1_vnum_u16(pred, p, 3);

       //load weight index, 1 means 1 vnum displacement from the base,
       // as we joinly store the indices and values of the sparse matrix
    svfloat16_t A_g_0 = svreinterpret_f16(
        svld1_gather_index(pred, (float64_t *)A, svreinterpret_u64(B_k_0)));
    svfloat16_t A_g_1 = svreinterpret_f16(
        svld1_gather_index(pred, (float64_t *)A, svreinterpret_u64(B_k_1)));
        //gather from dense activation vector
    acc = svmla_x(pred, acc, A_g_0, B_v_0); //perform mla
    acc1 = svmla_x(pred, acc1, A_g_1, B_v_1); //perform mla
  }
  acc = svadd_z(pred, acc, acc1);

#else
  for (const unsigned short *p = from; p != to; p+= step) {
    /*
    const float16_t *pp = (const float16_t *)p;
    std::cout << pp[0] << ":" << pp[1] << ":" << pp[2] << ":" << pp[3] << ":"
    << pp[4] << ":" << pp[5] << ":" << pp[6] << ":" << pp[7] << std::endl;
    const unsigned short *ppp = &p[8];
    std::cout << ppp[0] << ":" << ppp[1] << ":" << ppp[2] << ":"
    << ppp[3] << ":"
    << ppp[4] << ":" << ppp[5] << ":" << ppp[6] << ":" << ppp[7] << std::endl;
    */
    svuint16_t B_v_u = svld1_vnum_u16(pred, p, 0);
       //load weight value, 0 means no displacement from the base
    svfloat16_t B_v = svreinterpret_f16(B_v_u);
    svuint16_t B_k = svld1_vnum_u16(pred, p, 1);
       //load weight index, 1 means 1 vnum displacement from the base,
       // as we joinly store the indices and values of the sparse matrix
    svfloat16_t A_g = svreinterpret_f16(
        svld1_gather_index(pred, (float64_t *)A, svreinterpret_u64(B_k)));
        //gather from dense activation vector
    acc = svmla_x(pred, acc, A_g, B_v); //perform mla

    // svst1(true32, &C[i*vitems], A_g);
  }
#endif

    if (reduction) {
      float16_t r = svaddv(pred, acc);
      *Out_addr = r;
    } else {
      svst1(pred, Out_addr, acc);
    }

}
inline void sp_row_joined_fused(const float16_t *A,
    const unsigned short *B_join,
    unsigned int start, unsigned int end, unsigned int nitems,
    float16_t *Out_addr, bool reduction) {

  svfloat16_t acc = svdup_f16(0);
  const svbool_t pred = svptrue_b16();
  int step = 2 * nitems;
  const unsigned short * from = &B_join[start * step+nitems];
  const unsigned short * to = &B_join[end * step+nitems];

  for (const unsigned short *p = from; p != to; p+= step) {
    /*
    const float16_t *pp = (const float16_t *)p;
    std::cout << pp[0] << ":" << pp[1] << ":" << pp[2] << ":" << pp[3] << ":"
    << pp[4] << ":" << pp[5] << ":" << pp[6] << ":" << pp[7] << std::endl;
    const unsigned short *ppp = &p[8];
    std::cout << ppp[0] << ":" << ppp[1] << ":" << ppp[2] << ":"
    << ppp[3] << ":"
    << ppp[4] << ":" << ppp[5] << ":" << ppp[6] << ":" << ppp[7] << std::endl;
    */
      // std::cout<<"normal: base_1="<<p<<std::endl;
    svuint16_t B_v_u = svld1_vnum_u16(pred, p, -1);
    svfloat16_t B_v = svreinterpret_f16(B_v_u);
    svfloat16_t A_g = spcpuFusedGather_f16(pred,p,A);

    acc = svmla_x(pred, acc, A_g, B_v); //perform mla
  }
  if (reduction) {
    float16_t r = svaddv(pred, acc);
    *Out_addr = r;
  } else {
      svst1(pred, Out_addr, acc);
    }

}
#endif

/*
  The size of matrix B is K * N, but it is transposed to N * K
  Use CSR type of encoding
*/

void spmm_scalar(float16_t *C, const float16_t *A,
    const float16_t *B_value, const unsigned short *B_indices,
    const unsigned int *B_indptr,
    unsigned long M, unsigned long K,
    unsigned long N) {

  for (unsigned int i = 0; i < M; i++) {
    // std::cout << "i: " << i << std::endl;
    for (unsigned int j = 0; j < N; j++) {
      unsigned int start = B_indptr[j];
      unsigned int end = B_indptr[j+1];
      float16_t res = 0;
      for (unsigned int k = start; k < end; k++) {
        float16_t B_v = B_value[k];
        unsigned short B_k = B_indices[k];
        float16_t A_v = A[i * K + B_k];
        res += B_v * A_v;
      }
      C[i * N + j] = res;
    }
  }
}

/*
  Matrix A is transposed: (K, M)
*/
void spmm_irregular(float16_t *C, const float16_t *A,
    const float16_t *B_value, const unsigned short *B_indices,
    const unsigned int *B_indptr,
    unsigned long M, unsigned long K,
    unsigned long N) {
  /*
  const svbool_t pred = svptrue_b16();
  for (unsigned int i = 0; i < M; i++) {
    // std::cout << "i: " << i << std::endl;
    for (unsigned int j = 0; j < (unsigned)(N / block); j++) {
      // std::cout << "j: " << j << std::endl;
      unsigned int start = B_indptr[j];
      unsigned int end = B_indptr[j+1];
      svfloat16_t res = svdup_f16(0);
      for (unsigned int k = start; k < end; k++) {
        // std::cout << "k: " << k << std::endl;
        const svfloat16_t B_v = svld1(pred, &B_value[k * block]);
        unsigned short B_k = B_indices[k];
        const svfloat16_t A_v = svdup_f16(A[i * K + B_k]);
        res = svmla_x(pred, res, A_v, B_v);
      }
      svst1(pred, &C[i * N + j * block], res);
    }
  }
  */
}

/*
  Block sparse format, the B_indptr saves the block number
  i.e. the max value is N / BLOCK. N must be divisable of BLOCK
  The tiling is not done
*/
void  spmm_block_vertical(float16_t *C, float16_t const *A,
    const float16_t *B_value, const unsigned short *B_indices,
    const unsigned int *B_indptr,
    unsigned long M, unsigned long K,
    unsigned long N, unsigned block) {

  assert(block == svcnth());
  const svbool_t pred = svptrue_b16();
  for (unsigned int i = 0; i < M; i++) {
    // std::cout << "i: " << i << std::endl;
    for (unsigned int j = 0; j < (unsigned)(N / block); j++) {
      // std::cout << "j: " << j << std::endl;
      unsigned int start = B_indptr[j];
      unsigned int end = B_indptr[j+1];
      svfloat16_t res = svdup_f16(0);
      for (unsigned int k = start; k < end; k++) {
        // std::cout << "k: " << k << std::endl;
        const svfloat16_t B_v = svld1(pred, &B_value[k * block]);
        unsigned short B_k = B_indices[k];
        const svfloat16_t A_v = svdup_f16(A[i * K + B_k]);
        res = svmla_x(pred, res, A_v, B_v);
      }
      svst1(pred, &C[i * N + j * block], res);
    }
  }
}

/*
  Block sparse format, the B_indptr saves the block number
  i.e. the max value is N / BLOCK. N must be divisable of BLOCK
  The tiling is not done
*/
void spmv_block_vertical(float32_t *C, float32_t const *A,
    const float16_t *B_value, const unsigned short *B_indices,
    const unsigned int *B_indptr,
    unsigned long K, unsigned long N, unsigned block) {

  assert(block == svcnth());
  const svbool_t pred16 = svptrue_b16();
  const svbool_t pred32 = svptrue_b32();
  const svbool_t pred16lo = svdupq_b16(false, false, false, false,
                                       true, true, true, true);
  const svbool_t pred16hi = svdupq_b16(true, true, true, true,
                                       false, false, false, false);
  // unsigned count = 0;
  for (unsigned int j = 0; j < (unsigned)(N / block); j++) {
    // std::cout << "j: " << j << std::endl;
    unsigned int start = B_indptr[j];
    unsigned int end = B_indptr[j+1];
    svfloat32_t res_lo = svdup_f32(0);
    svfloat32_t res_hi = svdup_f32(0);
    for (unsigned int k = start; k < end; k++) {
      // std::cout << "k: " << k << std::endl;
      const svfloat16_t B_v = svld1(pred16, &B_value[k * block]);
      const svfloat32_t B_v_hi = svcvt_f32_x(pred16hi, B_v);
      const svfloat32_t B_v_lo = svcvt_f32_x(pred16lo, B_v);
      unsigned short B_k = B_indices[k];
      const svfloat32_t A_v = svdup_f32(A[B_k]);
      res_lo = svmla_x(pred32, res_lo, A_v, B_v_lo);
      res_hi = svmla_x(pred32, res_hi, A_v, B_v_hi);
      // count += 8;
    }
    svst1_vnum(pred32, &C[j * block], 0, res_lo);
    svst1_vnum(pred32, &C[j * block], 1, res_hi);
  }
  // std::cout << count << std::endl;
}
/*
  HL: float32-version Block sparse format, the B_indptr saves the block number
  i.e. the max value is N / BLOCK. N must be divisable of BLOCK
  The tiling is not done
*/
void spmv_block_vertical_fp32(float32_t *C, float32_t const *A,
    const float32_t *B_value, const unsigned int *B_indices,
    const unsigned int *B_indptr,
    unsigned long K, unsigned long N, unsigned block) {
  std::cout<<"FP32 VERSION"<<std::endl;
  assert(block == svcntw());
  // const svbool_t pred16 = svptrue_b16();
  const svbool_t pred32 = svptrue_b32();
  // const svbool_t pred16lo = svdupq_b16(false, false, false, false,
  //                                      true, true, true, true);
  // const svbool_t pred16hi = svdupq_b16(true, true, true, true,
  //                                      false, false, false, false);
  // unsigned count = 0;
  for (unsigned int j = 0; j < (unsigned)(N / block); j++) {
    // std::cout << "j: " << j << std::endl;
    unsigned int start = B_indptr[j];
    unsigned int end = B_indptr[j+1];
    // svfloat32_t res_lo = svdup_f32(0);
    // svfloat32_t res_hi = svdup_f32(0);
    svfloat32_t res = svdup_f32(0);
    for (unsigned int k = start; k < end; k++) {
      // std::cout << "k: " << k << std::endl;
      const svfloat32_t B_v = svld1(pred32, &B_value[k * block]);
      // const svfloat32_t B_v_hi = svcvt_f32_x(pred16hi, B_v);
      // const svfloat32_t B_v_lo = svcvt_f32_x(pred16lo, B_v);
      unsigned int B_k = B_indices[k];
      const svfloat32_t A_v = svdup_f32(A[B_k]);
      // res_lo = svmla_x(pred32, res_lo, A_v, B_v_lo);
      // res_hi = svmla_x(pred32, res_hi, A_v, B_v_hi);
      res = svmla_x(pred32, res, A_v, B_v);
      // count += 8;
    }
    // svst1_vnum(pred32, &C[j * block], 0, res);
    // svst1_vnum(pred32, &C[j * block], 1, res_hi);
    svst1(pred32, &C[j*block],res);
  }
  // std::cout << count << std::endl;
}


/*
  Block sparse format, the B_indptr saves the block number
  i.e. the max value is N / BLOCK. N must be divisable of BLOCK
  The tiling is not done
*/
void spmv_block_vertical_split(float32_t *C, float32_t const *A,
    const float16_t *B_value, const unsigned short *B_indices,
    const unsigned int *B_indptr,
    unsigned long K, unsigned long N, unsigned block) {

  assert(block == svcnth());
  const svbool_t pred16 = svptrue_b16();
  const svbool_t pred32 = svptrue_b32();
  const svbool_t pred16lo = svdupq_b16(false, false, false, false,
                                       true, true, true, true);
  const svbool_t pred16hi = svdupq_b16(true, true, true, true,
                                       false, false, false, false);
  for (unsigned int j = 0; j < (unsigned)(N / block); j+=2) {
    // std::cout << "j: " << j << std::endl;
    unsigned int start0 = B_indptr[j];
    unsigned int end0 = B_indptr[j+1];
    unsigned int nnz0 = end0 - start0;
    unsigned int start1 = end0;
    unsigned int end1 = B_indptr[j+2];
    unsigned int nnz1 = end1 - start1;
    unsigned int lnnz = nnz1 > nnz0 ? nnz0 : nnz1;

    svfloat32_t acc0_lo = svdup_f32(0);
    svfloat32_t acc0_hi = svdup_f32(0);
    svfloat32_t acc1_lo = svdup_f32(0);
    svfloat32_t acc1_hi = svdup_f32(0);
    for (unsigned int k = 0; k < lnnz; k++) {
      // std::cout << "k: " << k << std::endl;
      const svfloat16_t B0_v = svld1(pred16, &B_value[start0 + k * block]);
      const svfloat32_t B0_v_hi = svcvt_f32_x(pred16hi, B0_v);
      const svfloat32_t B0_v_lo = svcvt_f32_x(pred16lo, B0_v);
      unsigned short B0_k = B_indices[start0 + k];
      const svfloat32_t A0_v = svdup_f32(A[B0_k]);
      acc0_lo = svmla_x(pred32, acc0_lo, A0_v, B0_v_lo);
      acc0_hi = svmla_x(pred32, acc0_hi, A0_v, B0_v_hi);

      const svfloat16_t B1_v = svld1(pred16, &B_value[start1 + k * block]);
      const svfloat32_t B1_v_hi = svcvt_f32_x(pred16hi, B1_v);
      const svfloat32_t B1_v_lo = svcvt_f32_x(pred16lo, B1_v);
      unsigned short B1_k = B_indices[start1 + k];
      const svfloat32_t A1_v = svdup_f32(A[B1_k]);
      acc1_lo = svmla_x(pred32, acc1_lo, A1_v, B1_v_lo);
      acc1_hi = svmla_x(pred32, acc1_hi, A1_v, B1_v_hi);
    }
    unsigned int start, rem_nnz;
    svfloat32_t acc_hi, acc_lo;
    if (lnnz < nnz0) {
      start = start0 + lnnz * block;
      rem_nnz = nnz0 - lnnz;
      acc_hi = acc0_hi;
      acc_lo = acc0_lo;
    } else {
      start = start1 + lnnz * block;
      rem_nnz = nnz1 - lnnz;
      acc_hi = acc1_hi;
      acc_lo = acc1_lo;
    }
    for (unsigned int k = 0; k < rem_nnz; k++) {
      const svfloat16_t B0_v = svld1(pred16, &B_value[start + k * block]);
      const svfloat32_t B0_v_hi = svcvt_f32_x(pred16hi, B0_v);
      const svfloat32_t B0_v_lo = svcvt_f32_x(pred16lo, B0_v);
      unsigned short B0_k = B_indices[start + k];
      const svfloat32_t A0_v = svdup_f32(A[B0_k]);
      acc_lo = svmla_x(pred32, acc_lo, A0_v, B0_v_lo);
      acc_hi = svmla_x(pred32, acc_hi, A0_v, B0_v_hi);
    }
    if (lnnz < nnz0) {
      acc0_lo = acc_lo;
      acc0_hi = acc_hi;
    } else {
      acc1_lo = acc_lo;
      acc1_hi = acc_hi;
    }
    svst1_vnum(pred32, &C[j * block], 0, acc0_lo);
    svst1_vnum(pred32, &C[j * block], 1, acc0_hi);
    svst1_vnum(pred32, &C[j * block], 2, acc1_lo);
    svst1_vnum(pred32, &C[j * block], 3, acc1_hi);
  }
}

void spmv_bucket_vertical(float32_t *C, const float32_t *A,
    const float16_t *B_value, const unsigned short *B_indices,
    const unsigned int *B_indptr,
    unsigned long K, unsigned long N, unsigned vitems) {

  assert(N % vitems == 0);
  // Only work on 128-bit chunks
  assert(vitems == 8);
  unsigned num_rows = (int)(N / vitems);
  const svbool_t pred16 = svptrue_b16();
  const svbool_t pred32 = svptrue_b32();
  const svbool_t pred16lo = svdupq_b16(false, false, false, false,
                                       true, true, true, true);
  const svbool_t pred16hi = svdupq_b16(true, true, true, true,
                                       false, false, false, false);
  svfloat32_t acc0 = svdup_f32(0);
  svfloat32_t acc1 = svdup_f32(0);
  for (unsigned int i = 0; i < num_rows; i++) {
    unsigned int start = B_indptr[i];
    unsigned int end = B_indptr[i+1];
    // std::cout << start << " : " << end << std::endl;
    for (unsigned int j = start; j < end; j++) {
      const svfloat16_t B_v = svld1(pred16, &B_value[j * vitems]);
      const svfloat32_t B_v_hi = svcvt_f32_x(pred16hi, B_v);
      const svfloat32_t B_v_lo = svcvt_f32_x(pred16lo, B_v);
      const svuint16_t B_k = svld1(pred16, &B_indices[j * vitems]);
      const svuint32_t B_k_hi = svunpkhi(B_k);
      const svuint32_t B_k_lo = svunpklo(B_k);
      const svfloat32_t A_lo = svld1_gather_index(pred32, A, B_k_lo);
      const svfloat32_t A_hi = svld1_gather_index(pred32, A, B_k_hi);
      acc0 = svmla_x(pred32, acc0, A_lo, B_v_lo);
      acc1 = svmla_x(pred32, acc1, A_hi, B_v_hi);
    }
    svst1_vnum(pred32, &C[i * vitems], 0, acc0);
    svst1_vnum(pred32, &C[i * vitems], 1, acc1);
  }
}


void spmv_bucket_vertical_split(float32_t *C, const float32_t *A,
    const float16_t *B_value, const unsigned short *B_indices,
    const unsigned int *B_indptr,
    unsigned long K, unsigned long N, unsigned vitems) {

  assert(N % vitems == 0);
  // Only work on 128-bit chunks
  assert(vitems == 8);
  unsigned num_rows = (int)(N / vitems);

  const svbool_t pred16 = svptrue_b16();
  const svbool_t pred32 = svptrue_b32();
  const svbool_t pred16lo = svdupq_b16(false, false, false, false,
                                       true, true, true, true);
  const svbool_t pred16hi = svdupq_b16(true, true, true, true,
                                       false, false, false, false);
  svfloat32_t acc0_hi = svdup_f32(0);
  svfloat32_t acc0_lo = svdup_f32(0);
  svfloat32_t acc1_hi = svdup_f32(0);
  svfloat32_t acc1_lo = svdup_f32(0);
  for (unsigned int i = 0; i < num_rows; i+=2) {
    unsigned int start0 = B_indptr[i];
    unsigned int end0 = B_indptr[i+1];
    unsigned int nnz0 = end0 - start0;
    unsigned int start1 = end0;
    unsigned int end1 = B_indptr[i+2];
    unsigned int nnz1 = end1 - start1;
    unsigned int lnnz = nnz1 > nnz0 ? nnz0 : nnz1;
    // std::cout << start0 << " : " << start1 << " : " << end1 << std::endl;
    // std::cout << nnz0 << " : " << nnz1 << std::endl;
    for (unsigned int j = 0; j < lnnz; j++) {
      const svfloat16_t B0_v = svld1(pred16, &B_value[start0 + j * vitems]);
      const svuint16_t B0_k = svld1(pred16, &B_indices[start0 + j * vitems]);
      const svfloat32_t B0_v_hi = svcvt_f32_x(pred16hi, B0_v);
      const svfloat32_t B0_v_lo = svcvt_f32_x(pred16lo, B0_v);
      const svuint32_t B0_k_hi = svunpkhi(B0_k);
      const svuint32_t B0_k_lo = svunpklo(B0_k);
      const svfloat32_t A0_lo = svld1_gather_index(pred32, A, B0_k_lo);
      const svfloat32_t A0_hi = svld1_gather_index(pred32, A, B0_k_hi);
      acc0_lo = svmla_x(pred32, acc0_lo, A0_lo, B0_v_lo);
      acc0_hi = svmla_x(pred32, acc0_hi, A0_hi, B0_v_hi);

      const svfloat16_t B1_v = svld1(pred16, &B_value[start1 + j * vitems]);
      const svuint16_t B1_k = svld1(pred16, &B_indices[start1 + j * vitems]);
      const svfloat32_t B1_v_hi = svcvt_f32_x(pred16hi, B1_v);
      const svfloat32_t B1_v_lo = svcvt_f32_x(pred16lo, B1_v);
      const svuint32_t B1_k_hi = svunpkhi(B1_k);
      const svuint32_t B1_k_lo = svunpklo(B1_k);
      const svfloat32_t A1_lo = svld1_gather_index(pred32, A, B1_k_lo);
      const svfloat32_t A1_hi = svld1_gather_index(pred32, A, B1_k_hi);
      acc1_lo = svmla_x(pred32, acc1_lo, A1_lo, B1_v_lo);
      acc1_hi = svmla_x(pred32, acc1_hi, A1_hi, B1_v_hi);
    }
    unsigned int start, rem_nnz;
    svfloat32_t acc_hi, acc_lo;
    if (lnnz < nnz0) {
      start = start0 + lnnz * vitems;
      rem_nnz = nnz0 - lnnz;
      acc_hi = acc0_hi;
      acc_lo = acc0_lo;
    } else {
      start = start1 + lnnz * vitems;
      rem_nnz = nnz1 - lnnz;
      acc_hi = acc1_hi;
      acc_lo = acc1_lo;
    }
    for (unsigned int j = 0; j < rem_nnz; j++) {
      const svfloat16_t B0_v = svld1(pred16, &B_value[start + j * vitems]);
      const svuint16_t B0_k = svld1(pred16, &B_indices[start + j * vitems]);
      const svfloat32_t B0_v_hi = svcvt_f32_x(pred16hi, B0_v);
      const svfloat32_t B0_v_lo = svcvt_f32_x(pred16lo, B0_v);
      const svuint32_t B0_k_hi = svunpkhi(B0_k);
      const svuint32_t B0_k_lo = svunpklo(B0_k);
      const svfloat32_t A0_lo = svld1_gather_index(pred32, A, B0_k_lo);
      const svfloat32_t A0_hi = svld1_gather_index(pred32, A, B0_k_hi);
      acc_lo = svmla_x(pred32, acc_lo, A0_lo, B0_v_lo);
      acc_hi = svmla_x(pred32, acc_hi, A0_hi, B0_v_hi);
    }
    if (lnnz < nnz0) {
      acc0_lo = acc_lo;
      acc0_hi = acc_hi;
    } else {
      acc1_lo = acc_lo;
      acc1_hi = acc_hi;
    }
    svst1(pred32, &C[i * vitems], acc0_lo);
    svst1(pred32, &C[(i+1) * vitems], acc0_hi);
    svst1(pred32, &C[(i+2) * vitems], acc1_lo);
    svst1(pred32, &C[(i+3) * vitems], acc1_hi);
  }
}


void spmv_bucket_vertical_split2(float32_t *C, const float32_t *A,
    const float16_t *B_value, const unsigned short *B_indices,
    const unsigned int *B_indptr,
    unsigned long K, unsigned long N, unsigned vitems) {

  assert(N % vitems == 0);
  // Only work on 128-bit chunks
  assert(vitems == 8);
  unsigned num_rows = (int)(N / vitems);
  const svbool_t true16 = svptrue_b16();
  const svbool_t true32 = svptrue_b32();
  const svbool_t true16lo = svdupq_b16(false, false, false, false,
                                       true, true, true, true);
  const svbool_t true16hi = svdupq_b16(true, true, true, true,
                                       false, false, false, false);
  svfloat32_t acc0_0 = svdup_f32(0);
  svfloat32_t acc1_0 = svdup_f32(0);
  svfloat32_t acc0_1 = svdup_f32(0);
  svfloat32_t acc1_1 = svdup_f32(0);
  for (unsigned int i = 0; i < num_rows; i++) {
    unsigned int start = B_indptr[i];
    unsigned int end = B_indptr[i+1];
    // std::cout << start << " : " << end << std::endl;
    for (unsigned int j = start; j < end; j+=2) {
      svfloat16_t B_v = svld1(true16, &B_value[j * vitems]);
      svfloat32_t B_v_hi = svcvt_f32_x(true16hi, B_v);
      svfloat32_t B_v_lo = svcvt_f32_x(true16lo, B_v);
      svuint16_t B_k = svld1(true16, &B_indices[j * vitems]);
      svuint32_t B_k_hi = svunpkhi(B_k);
      svuint32_t B_k_lo = svunpklo(B_k);
      svfloat32_t A_lo = svld1_gather_index(true32, A, B_k_lo);
      svfloat32_t A_hi = svld1_gather_index(true32, A, B_k_hi);
      acc0_0 = svmla_x(true32, acc0_0, A_lo, B_v_lo);
      acc1_0 = svmla_x(true32, acc1_0, A_hi, B_v_hi);

      svbool_t pred32 = svdup_b32(j+1 < end);
      svbool_t pred16 = svdup_b16(j+1 < end);
      B_v = svld1(pred16, &B_value[j * vitems]);
      B_v_hi = svcvt_f32_x(true16hi, B_v);
      B_v_lo = svcvt_f32_x(true16lo, B_v);
      B_k = svld1(pred16, &B_indices[j * vitems]);
      B_k_hi = svunpkhi(B_k);
      B_k_lo = svunpklo(B_k);
      A_lo = svld1_gather_index(pred32, A, B_k_lo);
      A_hi = svld1_gather_index(pred32, A, B_k_hi);
      acc0_1 = svmla_x(true32, acc0_1, A_lo, B_v_lo);
      acc1_1 = svmla_x(true32, acc1_1, A_hi, B_v_hi);
    }
    svfloat32_t acc0 = svadd_x(true32, acc0_0, acc0_1);
    svfloat32_t acc1 = svadd_x(true32, acc1_0, acc1_1);
    svst1_vnum(true32, &C[i * vitems], 0, acc0);
    svst1_vnum(true32, &C[i * vitems], 1, acc1);
  }
}

void spmv_joined_bucket_vertical(float32_t *C, const float32_t *A,
    const unsigned short *B_join,
    const unsigned int *B_indptr,
    unsigned long K, unsigned long N, unsigned vitems) {

  assert(N % vitems == 0);
  // Only work on 128-bit chunks
  assert(vitems == 8);
  unsigned num_rows = (int)(N / vitems);
  const svbool_t true16 = svptrue_b16();
  const svbool_t true32 = svptrue_b32();
  const svbool_t true16lo = svdupq_b16(false, false, false, false,
                                       true, true, true, true);
  const svbool_t true16hi = svdupq_b16(true, true, true, true,
                                       false, false, false, false);
  svfloat32_t acc0_0 = svdup_f32(0);
  svfloat32_t acc1_0 = svdup_f32(0);

  for (unsigned int i = 0; i < num_rows; i++) {
    unsigned int start = B_indptr[i];
    unsigned int end = B_indptr[i+1];
    // std::cout << start << " : " << end << std::endl;
    for (unsigned int j = start; j < end; j++) {
      svuint16_t B_v_u = svld1_vnum(true16, &B_join[j * 2 * vitems], 0);
      svfloat16_t B_v = svreinterpret_f16(B_v_u);
      svfloat32_t B_v_hi = svcvt_f32_x(true16hi, B_v);
      svfloat32_t B_v_lo = svcvt_f32_x(true16lo, B_v);
      svuint16_t B_k = svld1_vnum(true16, &B_join[j * 2 * vitems], 1);
      svuint32_t B_k_hi = svunpkhi(B_k);
      svuint32_t B_k_lo = svunpklo(B_k);
      /*
      svuint32_t B_k_lo = svld1uh_vnum_u32(true16lo,
                                           &B_join[j * 2 * vitems + vitems],
                                           0);
      svuint32_t B_k_hi = svld1uh_vnum_u32(true16hi,
                                           &B_join[j * 2 * vitems + vitems],
                                           svcntw());
      */
      svfloat32_t A_lo = svld1_gather_index(true32, A, B_k_lo);
      svfloat32_t A_hi = svld1_gather_index(true32, A, B_k_hi);
      acc0_0 = svmla_x(true32, acc0_0, A_lo, B_v_lo);
      acc1_0 = svmla_x(true32, acc1_0, A_hi, B_v_hi);
    }
    svst1_vnum(true32, &C[i * vitems], 0, acc0_0);
    svst1_vnum(true32, &C[i * vitems], 1, acc1_0);
  }
}



//float32 version of sparse matrix
void spmv_joined_bucket_vertical_sparsefp32(float32_t *C, const float32_t *A,
    const unsigned int *B_join,
    const unsigned int *B_indptr,
    unsigned long K, unsigned long N, unsigned vitems) {

  assert(N % vitems == 0);
  // Only work on 128-bit chunks
  assert(vitems == 4);
  unsigned num_rows = (int)(N / vitems);
  // const svbool_t true16 = svptrue_b16();
  const svbool_t true32 = svptrue_b32();
  // const svbool_t true32lo = svdupq_b32(false, false,
  //                                      true, true);
  // const svbool_t true32hi = svdupq_b32(true, true,
  //                                      false, false);
  // svfloat64_t acc0_0 = svdup_f64(0);
  // svfloat64_t acc1_0 = svdup_f64(0);
  svfloat32_t acc = svdup_f32(0);

  for (unsigned int i = 0; i < num_rows; i++) {
    unsigned int start = B_indptr[i];
    unsigned int end = B_indptr[i+1];
    // std::cout << start << " : " << end << std::endl;
    // float32_t *B_v_array = new float32_t[vitems];
    for (unsigned int j = start; j < end; j++) {
      svuint32_t B_v_u = svld1_vnum_u32(true32, &B_join[j * 2 * vitems], 0); //load weight value, 0 means no displacement from the base
      svfloat32_t B_v = svreinterpret_f32(B_v_u);
      // svst1(true32, B_v_array, B_v);
      //   for(unsigned int i=0; i<vitems; i++){
      //     std::cout<<"B_v_array["<<i<<"]="<<B_v_array[i]<<std::endl;
      // }
      // std::cout<<B_join[j*2*vitems]<<"\t"<<B_join[j*2*vitems+1]<<"\t"<<B_join[j*2*vitems+2]<<"\t"<<B_join[j*2*vitems+3]<<std::endl;
      // std::cout<<B_join[j*2*vitems+4]<<"\t"<<B_join[j*2*vitems+5]<<"\t"<<B_join[j*2*vitems+6]<<"\t"<<B_join[j*2*vitems+7]<<std::endl;
      svuint32_t B_k = svld1_vnum_u32(true32, &B_join[j * 2 * vitems], 1); //load weight index, 1 means 1 vnum displacement from the base, as we joinly store the indices and values of the sparse matrix
      svfloat32_t A_g = svld1_gather_index(true32, A, B_k); //gather from dense activation vector
      acc = svmla_x(true32, acc, A_g, B_v); //perform mla
      // svst1(true32, &C[i*vitems], A_g);
    }
    svst1(true32, &C[i*vitems], acc); // vector store in result vector C
  }
  // std::cout<<"Result vector:\n";
  // for(unsigned int i=0; i<N; i++){
  //   std::cout<<"c["<<i<<"]="<<C[i]<<std::endl;
  // }
}

//float16 version of sparse matrix
void spmv_joined_bucket_vertical_densefp16(float16_t *C, const float16_t *A,
    const unsigned short *B_join,
    const unsigned int *B_indptr,
    unsigned long K, unsigned long N, unsigned vitems) {

  assert(N % vitems == 0);
  // Only work on 128-bit chunks
   assert(vitems == 8);
  unsigned num_rows = (int)(N / vitems);

  // const svbool_t true8 = svptrue_pat_b8(SV_VL128);
  const svbool_t true16 = svptrue_b16();
  // const svbool_t true64 = svptrue_b64();

  svfloat16_t acc = svdup_f16(0);

  for (unsigned int i = 0; i < num_rows; i++) {
    unsigned int start = B_indptr[i];
    unsigned int end = B_indptr[i+1];
    /*
    acc = svdup_f16(0);
    // std::cout << start << " : " << end << std::endl;
    // float16_t *B_v_array = new float16_t[vitems];
    for (unsigned int j = start; j < end; j++) {
      svuint16_t B_v_u = svld1_vnum_u16(true16, &B_join[j * 2 * vitems], 0);
      svfloat16_t B_v = svreinterpret_f16(B_v_u);

      svuint16_t B_k = svld1_vnum_u16(true64, &B_join[j * 2 * vitems], 1);
                        //   for (int m = 0; m <vitems; m++){
              //std::cout <<"the loaded A:"<<B_k[m]<<std::endl;
          //}
         //load weight index, 1 means 1 vnum displacement from the base,
         // as we joinly store the indices and values of the sparse matrix
      svfloat16_t A_g = svreinterpret_f16(
          svld1_gather_index(true64, (float64_t *)A, svreinterpret_u64(B_k)));
          //gather from dense activation vector
      acc = svmla_x(true16, acc, A_g, B_v); //perform mla
      //svst1(true16, &C[i*vitems], A_g);
       // for (int m = 0; m <vitems; m++){
            //  std::cout <<"the loaded A:"<<C[i*vitems+m]<<std::endl;
        //  }
    }
    */
    sp_row_joined(A, B_join, start, end, vitems,
        &C[i*vitems], false);
    // svst1(true16, &C[i*vitems], acc); // vector store in result vector C
  }
}
// float16 version of sparse matrix
void spmv_joined_bucket_vertical_densefp16_fused(
    float16_t *C, const float16_t *A,
    const unsigned short *B_join,
    const unsigned int *B_indptr,
    unsigned long K, unsigned long N, unsigned vitems) {

  assert(N % vitems == 0);
  // Only work on 128-bit chunks
   assert(vitems == 8);
  unsigned num_rows = (int)(N / vitems);

  const svbool_t true16 = svptrue_b16();
  svfloat16_t acc = svdup_f16(0);

  for (unsigned int i = 0; i < num_rows; i++) {
    unsigned int start = B_indptr[i];
    unsigned int end = B_indptr[i+1];

    sp_row_joined_fused(A, B_join, start, end, vitems,
        &C[i*vitems], false);
  }
}

void spmv_joined_bucket_vertical_split(float32_t *C, const float32_t *A,
    const unsigned short *B_join,
    const unsigned int *B_indptr,
    unsigned long K, unsigned long N, unsigned vitems) {

  assert(N % vitems == 0);
  // Only work on 128-bit chunks
  assert(vitems == 8);
  unsigned num_rows = (int)(N / vitems);
  const svbool_t true16 = svptrue_b16();
  const svbool_t true32 = svptrue_b32();
  const svbool_t true16lo = svdupq_b16(false, false, false, false,
                                       true, true, true, true);
  const svbool_t true16hi = svdupq_b16(true, true, true, true,
                                       false, false, false, false);
  svfloat32_t acc0_0 = svdup_f32(0);
  svfloat32_t acc1_0 = svdup_f32(0);
  svfloat32_t acc0_1 = svdup_f32(0);
  svfloat32_t acc1_1 = svdup_f32(0);
  for (unsigned int i = 0; i < num_rows; i++) {
    unsigned int start = B_indptr[i];
    unsigned int end = B_indptr[i+1];
    // std::cout << start << " : " << end << std::endl;
    for (unsigned int j = start; j < end; j+=2) {
      svuint16_t B_v_u = svld1_vnum(true16, &B_join[j * 2 * vitems], 0);
      svfloat16_t B_v = svreinterpret_f16(B_v_u);
      svfloat32_t B_v_hi = svcvt_f32_x(true16hi, B_v);
      svfloat32_t B_v_lo = svcvt_f32_x(true16lo, B_v);

      svuint16_t B_k = svld1_vnum(true16, &B_join[j * 2 * vitems], 1);
      svuint32_t B_k_hi = svunpkhi(B_k);
      svuint32_t B_k_lo = svunpklo(B_k);
      /*
      svuint32_t B_k_lo = svld1uh_vnum_u32(true16lo,
                                           &B_join[j * 2 * vitems + vitems],
                                           0);
      svuint32_t B_k_hi = svld1uh_vnum_u32(true16hi,
                                           &B_join[j * 2 * vitems + vitems],
                                           svcntw());
      */
      svfloat32_t A_lo = svld1_gather_index(true32, A, B_k_lo);
      svfloat32_t A_hi = svld1_gather_index(true32, A, B_k_hi);
      acc0_0 = svmla_x(true32, acc0_0, A_lo, B_v_lo);
      acc1_0 = svmla_x(true32, acc1_0, A_hi, B_v_hi);

      svbool_t pred32 = svdup_b32(j+1 < end);
      svbool_t pred16 = svdup_b16(j+1 < end);
      B_v_u = svld1_vnum(pred16, &B_join[(j+1) * 2 * vitems], 0);
      B_v = svreinterpret_f16(B_v_u);
      B_v_hi = svcvt_f32_x(true16hi, B_v);
      B_v_lo = svcvt_f32_x(true16lo, B_v);
      B_k = svld1_vnum(pred16, &B_join[(j+1) * 2 * vitems], 1);
      B_k_hi = svunpkhi(B_k);
      B_k_lo = svunpklo(B_k);
      /*
      B_k_lo = svld1uh_vnum_u32(true16lo,
                                &B_join[(j+1) * 2 * vitems + vitems],
                                0);
      B_k_hi = svld1uh_vnum_u32(true16hi,
                                &B_join[(j+1) * 2 * vitems + vitems],
                                svcntw());
      */
      A_lo = svld1_gather_index(pred32, A, B_k_lo);
      A_hi = svld1_gather_index(pred32, A, B_k_hi);
      acc0_1 = svmla_x(true32, acc0_1, A_lo, B_v_lo);
      acc1_1 = svmla_x(true32, acc1_1, A_hi, B_v_hi);
    }
    svfloat32_t acc0 = svadd_x(true32, acc0_0, acc0_1);
    svfloat32_t acc1 = svadd_x(true32, acc1_0, acc1_1);
    svst1_vnum(true32, &C[i * vitems], 0, acc0);
    svst1_vnum(true32, &C[i * vitems], 1, acc1);
  }
}


void spmv_bucket_vertical_fp16(float16_t *C, const float16_t *A,
    const float16_t *B_value, const unsigned short *B_indices,
    const unsigned int *B_indptr,
    unsigned long K, unsigned long N, unsigned vitems) {

  // assert(N % vitems == 0);
  // Only work on 128-bit chunks
  // assert(vitems == 16);
  unsigned num_rows = (int)(N / vitems);
  const svbool_t pred16 = svptrue_b16();
  //const svbool_t pred32 = svptrue_b32();
  const svbool_t true64 = svptrue_b64();
  //const svbool_t predvlo= svwhilelt_b16(0,7);
  //const svbool_t predvhi = svnot_z(pred16, predvlo);
  //const svbool_t pred16lo = svdupq_b16(false, false, false, false,
   //                                    true, true, true, true);
  //const svbool_t pred16hi = svdupq_b16(true, true, true, true,
                                    //   false, false, false, false);


  svfloat16_t acc0 = svdup_f16(0);
  //svfloat32_t acc1 = svdup_f32(0);
  for (unsigned int i = 0; i < num_rows; i++) {
    unsigned int start = B_indptr[i];
    unsigned int end = B_indptr[i+1];
    // std::cout << start << " : " << end << std::endl;
    for (unsigned int j = start; j < end; j++) {
      const svfloat16_t B_v = svld1(pred16, &B_value[j * vitems]);
     // const svfloat32_t B_v_hi = svcvt_f32_x(predvlo, B_v);
      //const svfloat32_t B_v_lo = svcvt_f32_x(predvhi, B_v);
      const svuint16_t B_k = svld1(pred16, &B_indices[j * vitems]);
      //const svuint32_t B_k_hi = svunpkhi(B_k);
      //const svuint32_t B_k_lo = svunpklo(B_k);
      svfloat16_t A_g = svreinterpret_f16(svld1_gather_index(
          true64, (float64_t *)A, svreinterpret_u64(B_k)));
      //const svfloat32_t A_lo = svld1_gather_index(pred32, A, B_k_lo);
      //const svfloat32_t A_hi = svld1_gather_index(pred32, A, B_k_hi);
      acc0 = svmla_x(pred16, acc0, A_g, B_v);
      //acc1 = svmla_x(pred32, acc1, A_hi, B_v_hi);
    }
    svst1(pred16, &C[i * vitems], acc0);
    //svst1_vnum(pred32, &C[i * vitems], 1, acc1);
  }
}

#if COMPUTE_FP32
void spmv_block_vertical_fp16(float16_t *C, float16_t const *A,
    const float16_t *B_value, const unsigned short *B_indices,
    const unsigned int *B_indptr,
    unsigned long K, unsigned long N, unsigned block) {

  assert(block == svcnth());
  const svbool_t pred16 = svptrue_b16();
  const svbool_t pred32 = svptrue_b32();
  const svbool_t pred16lo = svdupq_b16(false, false, false, false,
                                          true, true, true, true);
  const svbool_t pred16hi = svdupq_b16(true, true, true, true,
                                          false, false, false, false);

  for (unsigned int j = 0; j < (unsigned)(N / block); j++) {
    unsigned int start = B_indptr[j];
    unsigned int end = B_indptr[j+1];
    svfloat32_t res = svdup_f32(0);
    svfloat32_t res1 = svdup_f32(0);

    for (unsigned int k = start; k < end; k++) {
      const svfloat16_t B_v = svld1(pred16, &B_value[k * block]);
      svfloat16_t B_v_odd16 = svtrn2(B_v, B_v);
      const svfloat32_t B_v_even = svcvt_f32_x(pred16, B_v);
      const svfloat32_t B_v_odd = svcvt_f32_x(pred16, B_v_odd16);
    
      unsigned B_k = B_indices[k];
    //   std::cout << k << " : " << B_k << std::endl;
      const svfloat32_t A_v = svdup_f32(A[B_k]);

      res = svmla_x(pred32, res, A_v, B_v_even);
      res1 = svmla_x(pred32, res1, A_v, B_v_odd);
    }
    svfloat16_t acc16_odd = svcvt_f16_z(pred16, res1);
    svfloat16_t acc16_even = svcvt_f16_z(pred16, res);
    svfloat16_t acc16 = svtrn1(acc16_even, acc16_odd);
    svst1(pred16, &C[j * block], acc16);
  }
}

void spmv_block_horizonal_fp16(float16_t *C, float16_t const *A,
    const float16_t *B_value, const unsigned short *B_indices,
    const unsigned int *B_indptr,
    unsigned long K, unsigned long N, unsigned block) {

  assert(block == svcnth());
  const svbool_t pred16 = svptrue_b16();
  const svbool_t pred32 = svptrue_b16();
  const svbool_t pred16lo = svdupq_b16(false, false, false, false,
                                          true, true, true, true);
  const svbool_t pred16hi = svdupq_b16(true, true, true, true,
                                          false, false, false, false);
  for (unsigned int j = 0; j <N; j++) {
    unsigned int start = B_indptr[j];
    unsigned int end = B_indptr[j+1];
    svfloat32_t res = svdup_f32(0);
    svfloat32_t res1 = svdup_f32(0);

    for (unsigned int k = start; k < end; k++) {
      const svfloat16_t B_v = svld1(pred16, &B_value[k * block]);
      svfloat16_t B_v_odd16 = svtrn2(B_v, B_v);
      const svfloat32_t B_v_even = svcvt_f32_x(pred16, B_v);
      const svfloat32_t B_v_odd = svcvt_f32_x(pred16, B_v_odd16);
      unsigned short B_k = B_indices[k];
    //   std::cout << k << " : " << B_k << std::endl;
    //   const svfloat16_t A_v = svdup_f16(A[B_k]);
      const svfloat16_t A_v = svld1(pred16, &A[B_k * block]);
      svfloat16_t A_v_odd16 = svtrn2(A_v, A_v);
      const svfloat32_t A_v_even = svcvt_f32_x(pred16, A_v);
      const svfloat32_t A_v_odd = svcvt_f32_x(pred16, A_v_odd16);
      res = svmla_x(pred32, res, A_v_even, B_v_even);
      res1 = svmla_x(pred32, res1, A_v_odd, B_v_odd);
    }
    res = svadd_z(pred32, res, res1);
    float32_t r = svaddv(pred32, res);
    C[j] = r;
    //svst1(pred16, &C[j * block], res);
  }
}


#else
void spmv_block_vertical_fp16(float16_t *C, float16_t const *A,
    const float16_t *B_value, const unsigned short *B_indices,
    const unsigned int *B_indptr,
    unsigned long K, unsigned long N, unsigned block) {

  assert(block == svcnth());
  const svbool_t pred16 = svptrue_b16();

  for (unsigned int j = 0; j < (unsigned)(N / block); j++) {
    unsigned int start = B_indptr[j];
    unsigned int end = B_indptr[j+1];
    svfloat16_t res = svdup_f16(0);
    for (unsigned int k = start; k < end; k++) {
      const svfloat16_t B_v = svld1(pred16, &B_value[k * block]);
      unsigned short B_k = B_indices[k];
      // std::cout << k << " : " << B_k << std::endl;
      const svfloat16_t A_v = svdup_f16(A[B_k]);
      res = svmla_x(pred16, res, A_v, B_v);
    }
    svst1(pred16, &C[j * block], res);
  }
}

void spmv_block_horizonal_fp16(float16_t *C, float16_t const *A,
    const float16_t *B_value, const unsigned short *B_indices,
    const unsigned int *B_indptr,
    unsigned long K, unsigned long N, unsigned block) {

  assert(block == svcnth());
  const svbool_t pred16 = svptrue_b16();

  for (unsigned int j = 0; j <N; j++) {
    unsigned int start = B_indptr[j];
    unsigned int end = B_indptr[j+1];
    svfloat16_t res = svdup_f16(0);
    for (unsigned int k = start; k < end; k++) {
      const svfloat16_t B_v = svld1(pred16, &B_value[k * block]);
      unsigned short B_k = B_indices[k];
    //   std::cout << k << " : " << B_k << std::endl;
      //const svfloat16_t A_v = svdup_f16(A[B_k]);
      const svfloat16_t A_v = svld1(pred16, &A[B_k * block]);
      res = svmla_x(pred16, res, A_v, B_v);
    }
    float16_t acc0_sum = svaddv(pred16, res); //reduction
    C[j] = acc0_sum;
    //svst1(pred16, &C[j * block], res);
  }
}

#endif








void spmv_sve_fp16(float16_t *C, const float16_t *A, const float16_t *B,
              unsigned long K, unsigned long N) {
  //assert (N % (2 * svcnth()) == 0);
  //assert (K % svcntw() == 0);
  assert (svcnth() == 16);
  float16_t acc0_sum=0;
 // const svbool_t pred16lo = svdupq_b16(false, false, false, false,
  //                                     true, true, true, true);
  //const svbool_t pred16hi = svdupq_b16(true, true, true, true,
  //                                     false, false, false, false);
  const svbool_t pred16 = svptrue_b16();
  //const svbool_t pred32 = svptrue_b32();

  for (unsigned i = 0; i < N; i ++ ) {
    svfloat16_t acc0_0 = svdup_f16(0);
  //  svfloat32_t acc1 = svdup_f32(0);
    //svfloat32_t acc2 = svdup_f32(0);
    //svfloat32_t acc3 = svdup_f32(0);
    for (unsigned j = 0; j < K; j+= svcnth()) {
      const svfloat16_t B_v = svld1(pred16, &B[svcnth() * j + i*K]);
      //const svfloat32_t B_v0_hi = svcvt_f32_x(pred16hi, B_v0);
      //const svfloat32_t B_v0_lo = svcvt_f32_x(pred16lo, B_v0);
      //const svfloat16_t B_v1 = svld1_vnum(pred16, &B[N * j + i], 1);
      //const svfloat32_t B_v1_hi = svcvt_f32_x(pred16hi, B_v1);
      //const svfloat32_t B_v1_lo = svcvt_f32_x(pred16lo, B_v1);
      const svfloat16_t A_v = svld1(pred16, &A[j]);
      acc0_0 = svmla_x(pred16 , acc0_0, A_v, B_v);

      //acc0 = svmla_lane(acc0, B_v0_lo, A_v, 0);
      //acc1 = svmla_lane(acc1, B_v0_hi, A_v, 1);
      //acc2 = svmla_lane(acc2, B_v1_lo, A_v, 2);
     // acc3 = svmla_lane(acc3, B_v1_hi, A_v, 3);
    }
    acc0_sum = svaddv(pred16 , acc0_0);
     C[i] = acc0_sum;
    //svst1(pred16, &C[i], acc0_sum );
    //svst1(pred32, &C[i + svcntw()], acc1);
    //svst1(pred32, &C[i + 2 * svcntw()], acc2);
    //svst1(pred32, &C[i + 3 * svcntw()], acc3);
  }
}


// void sp_conv_horizontal_fp16(float16_t *C, float16_t const *A_padded,
//                   unsigned short const *W_join,
//                   const unsigned int *W_indptr,
//                   int A_h, int A_w, int W_h, int W_w, int C_in,
//                   unsigned int C_out,
//                   int nitems) {
//   unsigned int H_out = A_h - W_h + 1;
//   unsigned int W_out = A_w - W_w + 1;
//   const svbool_t pred16 = svptrue_b16();
//   for (unsigned int co = 0; co < C_out; co++) {
//     unsigned int start = W_indptr[co];
//     unsigned int end = W_indptr[co+1];
//     // std::cout << start << " : " << end << std::endl;
//     for (unsigned int h = 0; h < H_out; h++) {
//       for (unsigned int w = 0; w < W_out; w++) {
//         svfloat16_t res;
//         float16_t const * A_addr = &A_padded[(h * A_w + w) * C_in];
//         unsigned idx = (h * W_out + w) * C_out + co; // NHWC
//         sp_row_joined(A_addr, W_join, start, end, nitems,
//             &C[idx], true);
//         //float16_t r = svaddv(pred16, res);

//         //C[idx] = r;
//        // std::cout <<" C index "<< idx<< std::endl;
//       }
//     }
//   }

// }


// void sp_conv_vertical_fp16(float16_t *C, float16_t const *A_padded,
//                   unsigned short const *W_join,
//                   const unsigned int *W_indptr,
//                   int A_h, int A_w, int W_h, int W_w, int C_in,
//                   unsigned int C_out,
//                   int nitems) {
//   unsigned int H_out = A_h - W_h + 1;
//   unsigned int W_out = A_w - W_w + 1;
//   const svbool_t pred16 = svptrue_b16();
//   for (unsigned int co = 0; co < C_out/nitems; co++) {
//     unsigned int start = W_indptr[co];
//     unsigned int end = W_indptr[co+1];
//     // std::cout << start << " : " << end << std::endl;
//     for (unsigned int h = 0; h < H_out; h++) {
//       for (unsigned int w = 0; w < W_out; w++) {
//         svfloat16_t res;
//         float16_t const * A_addr = &A_padded[h * A_w + w];
//         unsigned idx = (h * W_out + w) * C_out+ co*nitems; // NHWC
//         sp_row_joined(A_addr, W_join, start, end, nitems,
//             &C[idx], false);
//        // float16_t r = svaddv(pred16, res);

//         // std::cout <<" C index "<< idx<< std::endl;
//        // svst1(pred16, &C[idx], res);
//        // C[idx] = r;
//        // std::cout <<" C index "<< idx<< std::endl;
//       }
//     }
//   }

// }


void spmv_joined_bucket_horizontal_fp16(float16_t *C, const float16_t *A,
    const unsigned short *B_join,
    const unsigned int *B_indptr,
    unsigned long K, unsigned long N, unsigned vitems) {
  svfloat16_t acc =svdup_f16(0);

  assert(N % vitems == 0);
  // Only work on 128-bit chunks
  assert(vitems == vitems);

  // svfloat16_t acc0_0 = svdup_f16(0);

  const svbool_t true16 = svptrue_b16();
  for (unsigned int i = 0; i < N; i++) {
    unsigned int start = B_indptr[i];
    unsigned int end = B_indptr[i+1];
    sp_row_joined(A,B_join, start, end, vitems, &C[i], true);
    //float16_t acc0_sum = svaddv(true16, acc); //reduction
    //C[i] = acc0_sum;
  }
}

inline svfloat16_t sp_row_joined_spilt(const float16_t *A,
    const unsigned short *B_join,
    unsigned int start, unsigned int end, unsigned int nitems,
    const svbool_t pred)
    {

      svfloat16_t acc0 = svdup_f16(0);
      svfloat16_t acc1 = svdup_f16(0);

    if(((end-start)%2) == 0){
      int step = 4*nitems;
      const unsigned short * from = &B_join[start*step];
      const unsigned short * to = &B_join[end*step];
      for (const unsigned short *p = from; p != to; p+=step){
        svuint16_t B0_v_u = svld1_vnum_u16(pred, p, 0);
        svfloat16_t B0_v = svreinterpret_f16(B0_v_u);
        svuint16_t B0_k = svld1_vnum_u16(pred, p, 1);

        svfloat16_t A0_g = svreinterpret_f16(
        svld1_gather_index(pred, (float64_t *)A, svreinterpret_u64(B0_k)));
        acc0 = svmla_x(pred, acc0, A0_g, B0_v);
        svuint16_t B1_v_u = svld1_vnum_u16(pred, p, 2);
        svfloat16_t B1_v = svreinterpret_f16(B1_v_u);
        svuint16_t B1_k = svld1_vnum_u16(pred, p, 3);
        svfloat16_t A1_g = svreinterpret_f16(
        svld1_gather_index(pred, (float64_t *)A, svreinterpret_u64(B1_k)));
        acc1 = svmla_x(pred, acc1, A1_g, B1_v);
        acc0 = svreinterpret_f16(svadd_u16_m(pred, svreinterpret_u16(acc0),svreinterpret_u16(acc1)));

      }
       //svst1_vnum_f16(pred, acc, 0, acc0);
       //svst1_vnum_f16(pred, acc, 1, acc1);
      //return std::make_pair(acc0,acc1);
      //return acc0;
    }
    else{
      int step = 2*nitems;
      const unsigned short * from = &B_join[start*step];
      const unsigned short * to = &B_join[end*step];
      for (const unsigned short *p = from; p != to; p+=step){
        svuint16_t B0_v_u = svld1_vnum_u16(pred, p, 0);
        svfloat16_t B0_v = svreinterpret_f16(B0_v_u);
        svuint16_t B0_k = svld1_vnum_u16(pred, p, 1);
        svfloat16_t A0_g = svreinterpret_f16(
        svld1_gather_index(pred, (float64_t *)A, svreinterpret_u64(B0_k)));
        acc0 = svmla_x(pred, acc0, A0_g, B0_v);
      }

    }
    return acc0;
    }



void spmv_joined_bucket_vertical_split_fp16(float16_t *C, const float16_t *A,
    const unsigned short *B_join,
    const unsigned int *B_indptr,
    unsigned long K, unsigned long N, unsigned vitems) {

  //assert(N % vitems == 0);
  // Only work on 128-bit chunks
  assert(vitems == 8);
  unsigned num_rows = (int)(N / vitems);

  const svbool_t true16 = svptrue_b16();
//const svbool_t true64 = svptrue_b64();
 // svfloat16_t n1= svdup_f16(0);
  //svfloat16_t n2= svdup_f16(0);
  svfloat16_t acc = svdup_f16(0);
 //float16_t c0,c1;
  //const svbool_t true32 = svptrue_b32();
  //const svbool_t true16lo = svdupq_b16(false, false, false, false,
   //                                    true, true, true, true);
  //const svbool_t true16hi = svdupq_b16(true, true, true, true,
  //                                     false, false, false, false);
 // svfloat16_t acc0_0 = svdup_f16(0);
  //svfloat16_t acc1_0 = svdup_f16(0);
  //svfloat16_t *acc = (svfloat16_t *)calloc(8, sizeof(float16_t));
  //svfloat32_t acc0_1 = svdup_f32(0);
  //svfloat32_t acc1_1 = svdup_f32(0);
  for (unsigned int i = 0; i < num_rows; i++) {
      unsigned int start = B_indptr[i];
      unsigned int end = B_indptr[i+1];
      acc= sp_row_joined_spilt(A, B_join, start, end, vitems,
          true16);
    // C[i]=acc[0];
    // n1 = svld1_vnum_f16(true16, acc,0);
    // c0 = svaddv(true16, n1); //reduction
     //n2 = svld1_vnum_f16(true16, acc,1);
     //c1 = svaddv(true16, n2); //reduction
     //C[i] = c0+c1;
    // std::cout <<" num_rows = "<< i<< std::endl;
    svst1(true16, &C[i * vitems], acc);
  }
}

#ifndef COMPUTE_FP32
void spmv_joined_bucket_vertical_scatter_fp16(float16_t *C, const float16_t *A,
    const unsigned short *B_join,
    const unsigned int *B_indptr,
    const unsigned short *B_rowind,
    unsigned long K, unsigned long N, unsigned vitems) {

  assert(N % vitems == 0);
  // Only work on 128-bit chunks
   assert(vitems == 8);
  unsigned num_rows = (int)(N / vitems);

  // const svbool_t true8 = svptrue_pat_b8(SV_VL128);
  const svbool_t true16 = svptrue_b16();
  // const svbool_t true64 = svptrue_b64();

  svfloat16_t acc = svdup_f16(0);

  for (unsigned int i = 0; i < num_rows; i++) {
    unsigned int start = B_indptr[i];
    unsigned int end = B_indptr[i+1];

    sp_row_joined_scatter(A, B_join, B_rowind, i*vitems, start, end, vitems,
        &C[i*vitems], false);
  }
}
#endif
