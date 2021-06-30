
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

#include <arm_sve.h>

void mm(float16_t *C, const float16_t *A, const float16_t *B,
    unsigned long M, unsigned long K,
    unsigned long N);

void mm_transposed(float16_t *C, const float16_t *A, const float16_t *B,
    unsigned long M, unsigned long K,
    unsigned long N);

void hgemm_01(float16_t *C, const float16_t *A, const float16_t *B,
    unsigned long M, unsigned long K,
    unsigned long N);

void hgemm_unrolled(float16_t *C, const float16_t *A, const float16_t *B,
    unsigned long M, unsigned long K,
    unsigned long N);

void hgemm_unrolled_split(float16_t *C, const float16_t *A, const float16_t *B,
    unsigned long M, unsigned long K, unsigned long N);

void mm_sve_split_fp16(float16_t *C, const float16_t *A, const float16_t *B,
              unsigned long K, unsigned long N);

void spmv_sve(float32_t *C, const float32_t *A, const float16_t *B,
              unsigned long K, unsigned long N);

void spmv_sve_fp16(float16_t *C, const float16_t *A, const float16_t *B,
              unsigned long K, unsigned long N);

void spmv_sve_split(float32_t *C, const float32_t *A, const float16_t *B,
              unsigned long K, unsigned long N);

void spmm_scalar(float16_t *C, const float16_t *A,
    const float16_t *B_value, const unsigned short *B_indices,
    const unsigned int *B_indptr,
    unsigned long M, unsigned long K,
    unsigned long N);

void spmm_block_vertical(float16_t *C, float16_t const *A,
    const float16_t *B_value, const unsigned short *B_indices,
    const unsigned int *B_indptr,
    unsigned long M, unsigned long K,
    unsigned long N, unsigned block);

void spmm_block_vertical_split(float16_t *C, const float16_t *A,
    const float16_t *B_value, const unsigned short *B_indices,
    const unsigned int *B_indptr,
    unsigned long M, unsigned long K,
    unsigned long N, unsigned block);

void spmm_block_vertical_split_transposed(float16_t *C, const float16_t *A,
    const float16_t *B_value, const unsigned short *B_indices,
    const unsigned int *B_indptr,
    unsigned long M, unsigned long K,
    unsigned long N, unsigned block);

void spmv_block_vertical(float32_t *C, float32_t const *A,
    const float16_t *B_value, const unsigned short *B_indices,
    const unsigned int *B_indptr,
    unsigned long K, unsigned long N, unsigned block);

void spmv_block_vertical_fp32(float32_t *C, float32_t const *A,
    const float32_t *B_value, const unsigned int *B_indices,
    const unsigned int *B_indptr,
    unsigned long K, unsigned long N, unsigned block);

void spmv_block_vertical_split(float32_t *C, float32_t const *A,
    const float16_t *B_value, const unsigned short *B_indices,
    const unsigned int *B_indptr,
    unsigned long K, unsigned long N, unsigned block);

void spmv_bucket_vertical(float32_t *C, const float32_t *A,
    const float16_t *B_value, const unsigned short *B_indices,
    const unsigned int *B_indptr,
    unsigned long K, unsigned long N, unsigned vitems);

void spmv_bucket_vertical_split(float32_t *C, const float32_t *A,
    const float16_t *B_value, const unsigned short *B_indices,
    const unsigned int *B_indptr,
    unsigned long K, unsigned long N, unsigned vitems);

void spmv_bucket_vertical_split2(float32_t *C, const float32_t *A,
    const float16_t *B_value, const unsigned short *B_indices,
    const unsigned int *B_indptr,
    unsigned long K, unsigned long N, unsigned vitems);

void spmv_joined_bucket_vertical(float32_t *C, const float32_t *A,
    const unsigned short *B_join,
    const unsigned int *B_indptr,
    unsigned long K, unsigned long N, unsigned vitems);

void spmv_joined_bucket_vertical_densefp16(float16_t *C, const float16_t *A,
    const unsigned short *B_join,
    const unsigned int *B_indptr,
    unsigned long K, unsigned long N, unsigned vitems);

void spmv_joined_bucket_vertical_densefp16_fused(
    float16_t *C, const float16_t *A,
    const unsigned short *B_join,
    const unsigned int *B_indptr,
    unsigned long K, unsigned long N, unsigned vitems);

void spmv_joined_bucket_vertical_split_fp16(float16_t *C, const float16_t *A,
    const unsigned short *B_join,
    const unsigned int *B_indptr,
    unsigned long K, unsigned long N, unsigned vitems);

void spmv_joined_bucket_vertical_sparsefp32(float32_t *C, const float32_t *A,
    const unsigned int *B_join,
    const unsigned int *B_indptr,
    unsigned long K, unsigned long N, unsigned vitems);

void spmv_joined_bucket_vertical_split(float32_t *C, const float32_t *A,
    const unsigned short *B_join,
    const unsigned int *B_indptr,
    unsigned long K, unsigned long N, unsigned vitems);

void spmv_block_vertical_fp16(float16_t *C, float16_t const *A,
    const float16_t *B_value, const unsigned short *B_indices,
    const unsigned int *B_indptr,
    unsigned long K, unsigned long N, unsigned block);

void spmv_bucket_vertical_fp16(float16_t *C, const float16_t *A,
    const float16_t *B_value, const unsigned short *B_indices,
    const unsigned int *B_indptr,
    unsigned long K, unsigned long N, unsigned vitems);

void sp_conv_horizontal_fp16(float16_t *C, float16_t const *A_padded,
                  unsigned short const *W_join,
                  const unsigned int *W_indptr,
                  int A_h, int A_w, int W_h, int W_w, int C_in,
                  unsigned int C_out,
                  int nitems);

void spmv_vertical_correct_old(float16_t *C, const float16_t *A,
                        const unsigned short *B_join,
                        const unsigned int *B_indptr,
                       unsigned long K, unsigned long N, unsigned vitem);

void spmv_vertical_correct(float16_t *C, const float16_t *A, const float16_t *B, 
                       unsigned long K, unsigned long N, unsigned M, unsigned vitems);

void spmv_horizontal_correct(float16_t *C, const float16_t *A, const float16_t *B, 
                       unsigned long K, unsigned long N, unsigned M, unsigned vitems);


void spmv_joined_bucket_horizontal_fp16(float16_t *C, const float16_t *A,
    const unsigned short *B_join,
    const unsigned int *B_indptr,
    unsigned long K, unsigned long N, unsigned vitems);

void spmv_block_horizonal_fp16(float16_t *C, float16_t const *A,
    const float16_t *B_value, const unsigned short *B_indices,
    const unsigned int *B_indptr,
    unsigned long K, unsigned long N, unsigned block);

void sp_conv_vertical_fp16(float16_t *C, float16_t const *A_padded,
                  unsigned short const *W_join,
                  const unsigned int *W_indptr,
                  int A_h, int A_w, int W_h, int W_w, int C_in,
                  unsigned int C_out,
                  int nitems);
void sp_conv_vertical_fp16_fused(float16_t *C, float16_t const *A_padded,
                  unsigned short const *W_join,
                  const unsigned int *W_indptr,
                  int A_h, int A_w, int W_h, int W_w, int C_in,
                  unsigned int C_out,
                  int nitems);
void dense_conv_fp16(float16_t *C, float16_t const *A_padded,
                  float16_t const *B,
                  int A_h, int A_w, unsigned W_h, unsigned W_w, unsigned C_in,
                  unsigned int C_out,
                  int nitems);

void sp_conv_horizontal_fp16_fp32_compute(
                  float16_t *C, float16_t const *A_padded,
                  unsigned short const *W_join,
                  const unsigned int *W_indptr,
                  int A_h, int A_w, int W_h, int W_w, int C_in,
                  unsigned int C_out,
                  int nitems);

void sp_conv_vertical_fp16_fp32_compute(
                  float16_t *C, float16_t const *A_padded,
                  unsigned short const *W_join,
                  const unsigned int *W_indptr,
                  int A_h, int A_w, int W_h, int W_w, int C_in,
                  unsigned int C_out,
                  int nitems);

void sp_conv_block_vertical_fp16_fp32_compute(
                  float16_t *C, float16_t const *A_padded,
                  const float16_t *W, const unsigned short *W_indices,
                  const unsigned int *W_indptr,
                  int A_h, int A_w, int W_h, int W_w, int C_in,
                  unsigned int C_out,
                  int nitems);

void sp_conv_block_horizontal_fp16_fp32_compute(
                  float16_t *C, float16_t const *A_padded,
                  const float16_t *W, const unsigned short *W_indices,
                  const unsigned int *W_indptr,
                  int A_h, int A_w, int W_h, int W_w, int C_in,
                  unsigned int C_out,
                  int nitems);

                  
void sp_conv_vertical_fp16(
                  float16_t *C, float16_t const *A_padded,
                  unsigned short const *W_join,
                  const unsigned int *W_indptr,
                  int A_h, int A_w, int W_h, int W_w, int C_in,
                  unsigned int C_out,
                  int nitems);
void sp_conv_vertical_fp16_fused(
                  float16_t *C, float16_t const *A_padded,
                  unsigned short const *W_join,
                  const unsigned int *W_indptr,
                  int A_h, int A_w, int W_h, int W_w, int C_in,
                  unsigned int C_out,
                  int nitems);
void sp_conv_block_vertical_fp16(
                  float16_t *C, float16_t const *A_padded,
                  const float16_t *W,
                  const unsigned short *W_indices,
                  const unsigned int *W_indptr,
                  int A_h, int A_w, int W_h, int W_w, int C_in,
                  unsigned int C_out,
                  int nitems);

void sp_conv_block_horizontal_fp16(
                  float16_t *C, float16_t const *A_padded,
                  const float16_t *W,
                  const unsigned short *W_indices,
                  const unsigned int *W_indptr,
                  int A_h, int A_w, int W_h, int W_w, int C_in,
                  unsigned int C_out,
                  int nitems);


void spmv_joined_bucket_vertical_scatter_fp16(float16_t *C, const float16_t *A,
    const unsigned short *B_join,
    const unsigned int *B_indptr,
    const unsigned short *B_rowind,
    unsigned long K, unsigned long N, unsigned vitems);


void sp_conv_vertical_fp16_fp32_compute_ref(float16_t *C, float16_t const *A, float16_t const *B,
                  int A_h, int A_w, int W_h, int W_w, int C_in, unsigned int C_out,
                  int nitems);

void sp_conv_horizontal_fp16_fp32_compute_ref(float16_t *C, float16_t const *A, float16_t const *B,
                  int A_h, int A_w, int W_h, int W_w, int C_in, unsigned int C_out,
                  int nitems);

void sp_conv_vertical_fp16_ref(float16_t *C, float16_t const *A, float16_t const *B,
                  int A_h, int A_w, int W_h, int W_w, int C_in, unsigned int C_out,
                  int nitems);

void sp_conv_horizontal_fp16_ref(float16_t *C, float16_t const *A, float16_t const *B,
                  int A_h, int A_w, int W_h, int W_w, int C_in, unsigned int C_out,
                  int nitems);

void sp_conv_vertical_fp16_fp32_compute_test(
                  float16_t *C, float16_t const *A_padded,
                  unsigned short const *W_join,
                  const unsigned int *W_indptr,
                  int A_h, int A_w, int W_h, int W_w, int C_in,
                  unsigned int C_out,
                  int nitems);

void dense_conv_ref(float16_t *C, float16_t const *A, float16_t const *B,
                  int A_h, int A_w, int W_h, int W_w, int C_in, unsigned int C_out,
                  int nitems);
