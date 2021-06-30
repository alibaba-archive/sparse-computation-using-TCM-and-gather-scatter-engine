'''
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
 */
'''

#include <cassert>
#include "mm.h"
#include <iostream>
/*
  The size of matrix B is K * N, but it is NOT transposed
*/

void mm(float16_t *C, float16_t const *A, float16_t const *B,
    unsigned long M, unsigned long K,
    unsigned long N) {
  for (unsigned long i = 0; i < M; ++i) {
    for (unsigned long j = 0; j < N; ++j) {
      C[i * N + j] = 0;
      for (unsigned long k = 0; k < K; ++k) {
        C[i * N + j] += A[i * K + k] * B[k * N + j];
      }
    }
  }
}

void hgemm_01(float16_t *C, float16_t const *A, float16_t const *B,
    unsigned long M, unsigned long K,
    unsigned long N) {
  for (unsigned long i = 0; i < M; ++i) {
    for (unsigned long j = 0; j < N; j += svcnth()) {
      svfloat16_t Acc = svdup_f16(0);
      const svbool_t pred_j = svwhilelt_b16(j, N);
      for (unsigned long k = 0; k < K; ++k) {
        const svfloat16_t A_i_k = svdup_f16(A[i * K + k]);
        const svfloat16_t B_k_j = svld1(pred_j, &B[k * N + j]);
        Acc = svmla_x(pred_j, Acc, A_i_k, B_k_j);
      }
      svst1(pred_j, &C[i * N + j], Acc);
    }
  }
}


void hgemm_unrolled(float16_t *C, float16_t const *A, float16_t const *B,
    unsigned long M, unsigned long K,
    unsigned long N) {
  const svbool_t all_active = svptrue_b16();

  for (unsigned long i = 0; i < M; ++i) {
    for (unsigned long j = 0; j < N; j += 2 * svcnth()) {
      svfloat16_t Acc_0 = svdup_f16(0);
      svfloat16_t Acc_1 = svdup_f16(0);
      const svbool_t pred_j_1 = svwhilelt_b16(j + svcnth(), N);
      const svbool_t pred_j_0 = svptest_first(all_active, pred_j_1)
      ? all_active : svwhilelt_b16(j, N);
      for (unsigned long k = 0; k < K; ++k) {
        const svfloat16_t A_i_k = svdup_f16(A[i * K + k]);
        const svfloat16_t B_k_j_0 = svld1_vnum(pred_j_0, &B[k * N + j], 0);
        const svfloat16_t B_k_j_1 = svld1_vnum(pred_j_1, &B[k * N + j], 1);
        Acc_0 = svmla_x(pred_j_0, Acc_0, A_i_k, B_k_j_0);
        Acc_1 = svmla_x(pred_j_1, Acc_1, A_i_k, B_k_j_1);
      }
      svst1_vnum(pred_j_0, &C[i * N + j], 0, Acc_0);
      svst1_vnum(pred_j_1, &C[i * N + j], 1, Acc_1);
    }
  }
}


void mm_transposed(float16_t *C, float16_t const *A, float16_t const *B,
  unsigned long M, unsigned long K,
  unsigned long N) {
  for (unsigned long i = 0; i < M; ++i) {
    for (unsigned long j = 0; j < N; ++j) {
      C[i * N + j] = 0;
      for (unsigned long k = 0; k < K; ++k) {
        C[i * N + j] += A[i * K + k] * B[j * N + k];
      }
    }
  }
}

/*
  Matrix B is (K, N)
*/
void spmv_sve(float32_t *C, const float32_t *A, const float16_t *B,
              unsigned long K, unsigned long N) {
  assert (N % (2 * svcnth()) == 0);
  assert (K % svcntw() == 0);
  assert (svcnth() == 8);
  const svbool_t pred16lo = svdupq_b16(false, false, false, false,
                                       true, true, true, true);
  const svbool_t pred16hi = svdupq_b16(true, true, true, true,
                                       false, false, false, false);
  const svbool_t pred16 = svptrue_b16();
  const svbool_t pred32 = svptrue_b32();

  for (unsigned i = 0; i < N; i += 2 * svcnth()) {
    svfloat32_t acc0 = svdup_f32(0);
    svfloat32_t acc1 = svdup_f32(0);
    svfloat32_t acc2 = svdup_f32(0);
    svfloat32_t acc3 = svdup_f32(0);
    for (unsigned j = 0; j < K; j+= svcntw()) {
      const svfloat16_t B_v0 = svld1(pred16, &B[N * j + i]);
      const svfloat32_t B_v0_hi = svcvt_f32_x(pred16hi, B_v0);
      const svfloat32_t B_v0_lo = svcvt_f32_x(pred16lo, B_v0);
      const svfloat16_t B_v1 = svld1_vnum(pred16, &B[N * j + i], 1);
      const svfloat32_t B_v1_hi = svcvt_f32_x(pred16hi, B_v1);
      const svfloat32_t B_v1_lo = svcvt_f32_x(pred16lo, B_v1);
      const svfloat32_t A_v = svld1(pred32, &A[j]);
      acc0 = svmla_lane(acc0, B_v0_lo, A_v, 0);
      acc1 = svmla_lane(acc1, B_v0_hi, A_v, 1);
      acc2 = svmla_lane(acc2, B_v1_lo, A_v, 2);
      acc3 = svmla_lane(acc3, B_v1_hi, A_v, 3);
    }
    svst1(pred32, &C[i], acc0);
    svst1(pred32, &C[i + svcntw()], acc1);
    svst1(pred32, &C[i + 2 * svcntw()], acc2);
    svst1(pred32, &C[i + 3 * svcntw()], acc3);
  }
}

void spmv_sve_fp16(float16_t *C, const float16_t *A, const float16_t *B,
              unsigned long K, unsigned long N) {
  assert (N % (2 * svcnth()) == 0);
  assert (K % svcntw() == 0);
  assert (svcnth() == 8);
  // const svbool_t pred16lo = svdupq_b16(false, false, false, false,
  //                                     true, true, true, true);
  // const svbool_t pred16hi = svdupq_b16(true, true, true, true,
  //                                     false, false, false, false);
  const svbool_t pred16 = svptrue_b16();
  //const svbool_t pred32 = svptrue_b32();

  for (unsigned i = 0; i < N; i += 1) {
    svfloat16_t acc0 = svdup_f16(0);
   // svfloat32_t acc1 = svdup_f32(0);
    //svfloat32_t acc2 = svdup_f32(0);
    //svfloat32_t acc3 = svdup_f32(0);
    for (unsigned j = 0; j < K; j+= svcnth()) {
      const svfloat16_t B_v0 = svld1(pred16, &B[N * j + i]); 
      //move weight VL elements from B matrix K dimension
     // const svfloat32_t B_v0_hi = svcvt_f32_x(pred16hi, B_v0);
     // const svfloat32_t B_v0_lo = svcvt_f32_x(pred16lo, B_v0);
     // const svfloat16_t B_v1 = svld1_vnum(pred16, &B[N * j + i], 1);
     // const svfloat32_t B_v1_hi = svcvt_f32_x(pred16hi, B_v1);
     // const svfloat32_t B_v1_lo = svcvt_f32_x(pred16lo, B_v1);
      const svfloat16_t A_v = svld1(pred16, &A[j]);
      //move activation VL elements from A matrix from K dimension
      acc0 = svmla_x(pred16, acc0, B_v0, A_v);
      //fmla operation
     // acc1 = svmla_lane(acc1, B_v0_hi, A_v, 1);
      //acc2 = svmla_lane(acc2, B_v1_lo, A_v, 2);
      //acc3 = svmla_lane(acc3, B_v1_hi, A_v, 3);
    }
    svst1(pred16, &C[i], acc0);
   // svst1(pred32, &C[i + svcntw()], acc1);
   // svst1(pred32, &C[i + 2 * svcntw()], acc2);
    //svst1(pred32, &C[i + 3 * svcntw()], acc3);
  }
}

inline svfloat16_t dense_row_conv(const float16_t *A,
    const float16_t *B,
     unsigned int nitems,int W_h, int W_w, int C_in,
    const svbool_t pred) {
 //std::cout <<" W_w= "<< W_w << std::endl;
  svfloat16_t acc = svdup_f16(0);
  unsigned int B_W = W_h*W_w*C_in;
  //std::cout <<" B_W= "<< B_W << std::endl;
  //std::cout <<" W_h= "<< W_h << std::endl;
  //std::cout <<" W_w= "<< W_w << std::endl;
  //std::cout <<" C_in= "<< C_in << std::endl;
  for ( unsigned short j = 0; j < B_W; j+=nitems) {
    const svfloat16_t B_v = svld1(pred, &B[j*nitems]);
    const svfloat16_t A_v = svld1(pred, &A[j*nitems]);
       //load weight value, 0 means no displacement from the base
    //svfloat16_t B_v = svreinterpret_f16(B_v_u);
    //svuint16_t B_k = svld1_vnum_u16(pred, p, 1);
       //load weight index, 1 means 1 vnum displacement from the base,
       // as we joinly store the indices and values of the sparse matrix
    //svfloat16_t A_g = svreinterpret_f16(
      //  svld1_gather_index(pred, (float64_t *)A, svreinterpret_u64(B_k)));
        //gather from dense activation vector
    acc = svmla_x(pred, acc, A_v, B_v); //perform mla
    // svst1(true32, &C[i*vitems], A_g);
  }
  return acc;
}


/*
void dense_conv_vertical_fp16(float16_t *C, float16_t const *A_padded,
                  float16_t const *B,
                  int A_h, int A_w, int W_h, int W_w, int C_in,
                  unsigned int C_out,
                  int nitems) {
      
  unsigned int H_out = A_h - W_h + 1;
  unsigned int W_out = A_w - W_w + 1;
  const svbool_t pred16 = svptrue_b16();
  //std::cout <<" W_w= "<< W_w << std::endl;
  for (unsigned int co = 0; co < C_out/nitems; co++) {
   // unsigned int start = W_indptr[co];
   // unsigned int end = W_indptr[co+1];
   
    for (unsigned int h = 0; h < H_out; h++) {
      for (unsigned int w = 0; w < W_out; w++) {
        svfloat16_t res;
        float16_t const * A_addr = &A_padded[h * A_w + w];
       // std::cout <<" W_w= "<< W_w << std::endl;
        res = dense_row_conv(A_addr,B, nitems, W_h, W_w,  C_in,
                 pred16);
       // float16_t r = svaddv(pred16, res);
      unsigned idx = (h * W_out + w) * C_out+ co*nitems; // NHWC
        // std::cout <<" C index "<< idx<< std::endl;
       svst1(pred16, &C[idx], res);
       // C[idx] = r;
       // std::cout <<" C index "<< idx<< std::endl;
      }
    }
  }

}
*/







/*
void dense_conv_vertical_fp16(float16_t *C, float16_t const *A_padded,
                  float16_t const *B,
                  int A_h, int A_w, int W_h, int W_w, int C_in,
                  unsigned int C_out,
                  int nitems) {
      
  unsigned int H_out = A_h - W_h + 1;
  unsigned int W_out = A_w - W_w + 1;
  const svbool_t pred16 = svptrue_b16();
  //std::cout <<" W_w= "<< W_w << std::endl;
  for (unsigned int co = 0; co < C_out/nitems; co++) {
   // unsigned int start = W_indptr[co];
   // unsigned int end = W_indptr[co+1];
   
    for (unsigned int h = 0; h < H_out; h++) {
      for (unsigned int w = 0; w < W_out; w++) {
        svfloat16_t res;
        float16_t const * A_addr = &A_padded[h * A_w + w];
        svfloat16_t acc0 = svdup_f16(0);
        svfloat16_t acc1 = svdup_f16(0);
        svfloat16_t acc2 = svdup_f16(0);
        svfloat16_t acc3 = svdup_f16(0);
        svfloat16_t acc4 = svdup_f16(0);
        svfloat16_t acc5 = svdup_f16(0);
        svfloat16_t acc6 = svdup_f16(0);
        svfloat16_t acc7 = svdup_f16(0);
        unsigned int B_W = W_h*W_w*C_in;
        for ( unsigned short j = 0; j < B_W; j+=nitems) {
          const svfloat16_t A_v = svld1(pred16, &A_addr[j*nitems]);
          const svfloat16_t B_v0 = svld1(pred16, &B[co*B_W+j*nitems]);
          acc0 = svmla_x(pred16, acc0, A_v, B_v0); 

          const svfloat16_t B_v1 = svld1(pred16, &B[co*B_W+(j+1)*nitems]);
          acc1 = svmla_x(pred16, acc1, A_v, B_v1);
      
          const svfloat16_t B_v2 = svld1(pred16, &B[co*B_W+(j+2)*nitems]);
          acc2 = svmla_x(pred16, acc2, A_v, B_v2);  

          const svfloat16_t B_v3 = svld1(pred16, &B[co*B_W+(j+3)*nitems]);
          acc3 = svmla_x(pred16, acc3, A_v, B_v3);  

          const svfloat16_t B_v4 = svld1(pred16, &B[co*B_W+(j+4)*nitems]);
          acc4 = svmla_x(pred16, acc4, A_v, B_v4);
      
          const svfloat16_t B_v5 = svld1(pred16, &B[co*B_W+(j+5)*nitems]);
          acc5 = svmla_x(pred16, acc5, A_v, B_v5);  

          const svfloat16_t B_v6 = svld1(pred16, &B[co*B_W+(j+6)*nitems]);
          acc6 = svmla_x(pred16, acc6, A_v, B_v6);  

          const svfloat16_t B_v7 = svld1(pred16, &B[co*B_W+(j+7)*nitems]);
          acc7 = svmla_x(pred16, acc7, A_v, B_v7);   

        }
         svfloat16_t acc  =svadd_x(pred16, acc0, acc1);
          acc  =svadd_x(pred16, acc, acc2);
          acc  =svadd_x(pred16, acc, acc3);
          acc  =svadd_x(pred16, acc, acc4);
          acc  =svadd_x(pred16, acc, acc5);
          acc  =svadd_x(pred16, acc, acc6);
          acc  =svadd_x(pred16, acc, acc7);
       // float16_t r = svaddv(pred16, res);
       unsigned idx = (h * W_out + w) * C_out+ co*nitems; // NHWC
        // std::cout <<" C index "<< idx<< std::endl;
       svst1(pred16, &C[idx], acc);
       // C[idx] = r;
       // std::cout <<" C index "<< idx<< std::endl;
      }
    }
  }

}*/
