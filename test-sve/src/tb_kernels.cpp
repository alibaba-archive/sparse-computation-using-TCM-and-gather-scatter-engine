

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

#include "mm.h"

void spmv_vertical_correct_old(float16_t *C, const float16_t *A, const unsigned short *B_join, 
                        const unsigned int *B_indptr,
                       unsigned long K, unsigned long N, unsigned vitems)
                       {

         unsigned num_rows = (int)(N / vitems);
         float16_t *B_v = (float16_t *)calloc(vitems, sizeof(float16_t ));
         unsigned short *B_i = (unsigned short *)calloc(vitems, sizeof(unsigned short));
         float16_t *A_i = (float16_t *)calloc(vitems, sizeof(float16_t ));
         float16_t *acc0= (float16_t *)calloc(vitems, sizeof(float16_t ));

        unsigned short tmp = 0;
        for (unsigned int i = 0; i < num_rows; i++) {
               unsigned int start = B_indptr[i];
               unsigned int end = B_indptr[i+1];
               for (unsigned int j = start; j < end; j++) {
                      for(unsigned int m = 0; m<vitems; m++){
                       tmp = B_join[vitems*2*j+m]; 
                      // std::cout <<"the loaded weight:"<<vitems*2*j+m<<std::endl;
                         //B_v[m] = (*(float16_t *)&tmp);
                        B_v[m] = reinterpret_cast<float16_t &>(tmp);
                     // std::cout <<"the loaded weight:"<<B_v[m]<<std::endl;
                       B_i[m] = B_join[vitems*(2*j+1)+m]; 
                       //std::cout <<"the loaded index location:"<<B_i[m]<<std::endl;
                       A_i[m] = A[B_i[m]];          
                       acc0[m] = acc0[m] +  A_i[m]* B_v[m];              
                   }
               //for (int m = 0; m < vitems; m++){
                  
                 // std::cout <<"the loaded weight:"<<B_v[m]<<std::endl;
                  //std::cout <<"the loaded index:"<<B_i[m]<<std::endl;
                  //std::cout <<"the loaded activation:"<<A_i[m]<<std::endl;
               //   std::cout <<"the fmla results:"<<acc0[m]<<std::endl;
                // }                 
               }
               for (unsigned int n = 0; n < vitems; n++) {
                   C[i*vitems+ n] = acc0[n] ; 
                   acc0[n] = 0;           }
           }
           
    }

void spmv_vertical_correct(float16_t *C, const float16_t *A, const float16_t *B, 
                       unsigned long K, unsigned long N, unsigned M, unsigned vitems)
                       {
    for (unsigned int m = 0; m < M; m++){
        for (unsigned int k = 0; k < K; k++){
            float16_t acc = 0.0;
            for (unsigned int n = 0; n < N; n++){
                acc += A[N*m + n] * B[K*n + k];
            }
            C[K*m +k] = acc;
        }
    }
}
void spmv_horizontal_correct(float16_t *C, const float16_t *A, const float16_t *B, 
                       unsigned long K, unsigned long N, unsigned M, unsigned vitems)
                       {
    for (unsigned int m = 0; m < M; m++){
        for (unsigned int k = 0; k < K; k++){
            float16_t acc = 0.0;
            for (unsigned int n = 0; n < N; n++){
                acc += A[N*m + n] * B[K*n + k];
            }
            C[K*m +k] = acc;
        }
    }
}

    


/*void sp_conv_vertical_fp16_fp32_compute_ref(
                  float16_t *C, float16_t const *A_padded,
                  unsigned short const *W_join,
                  const unsigned int *W_indptr,
                  int A_h, int A_w, int W_h, int W_w, int C_in,
                  unsigned int C_out,
                  int nitems)  
                  {   

         float16_t *B_v = (float16_t *)calloc(nitems, sizeof(float16_t ));
         unsigned short *B_i = (unsigned short *)calloc(nitems, sizeof(unsigned short));
         float16_t *acc0= (float16_t *)calloc(nitems, sizeof(float16_t ));  
         float16_t *A_i = (float16_t *)calloc(nitems, sizeof(float16_t )); 
         unsigned short tmp = 0;
                    unsigned int H_out = A_h - W_h + 1;
                    unsigned int W_out = A_w - W_w + 1;
                    for (unsigned int co = 0; co < C_out/nitems; co++){
                      unsigned int start = W_indptr[co];
                      unsigned int end = W_indptr[co+1];
                        for (unsigned int h = 0; h < H_out; h++){
                           for (unsigned int w = 0; w < W_out; w++){
                               const float16_t *A_addr = &A_padded[h*A_w + w];
                                unsigned idx = (h * W_out + w) * C_out + co*nitems;
                                int step = 2 * nitems;
                                const unsigned short from = W_join[start * step];
                                const unsigned short to = W_join[end * step];
                                for (unsigned short p = from; p != to; p+= step){
                                for (unsigned int m = 0; m < nitems; m++){
                                  tmp = W_join[nitems*2*p+m];
                                  B_v[m] = reinterpret_cast<float16_t &>(tmp);
                                  B_i[m] = W_join[nitems*(2*p+1)+m]; 
                       std::cout <<"the loaded index location:"<<B_i[m]<<std::endl;
                                  A_i[m] = A_addr[B_i[m]];          
                                  acc0[m] = acc0[m] +  A_i[m]* B_v[m]; 
                                }
                                }
                                     for (unsigned int n = 0; n < nitems; n++) {
                                           C[idx+n] = acc0[n] ; 
                                           acc0[n] = 0;       }
                          }
                        }


                    }




                  }*/


inline unsigned int idx_A(unsigned int h, unsigned int w, unsigned int c,
                    unsigned int A_h, unsigned int A_w, unsigned int A_c){
    return h * (A_w * A_c) + w * A_c + c;
}

inline unsigned int idx_B(unsigned int h, unsigned int w, unsigned int c_in, unsigned int c_out,
                    unsigned int W_h, unsigned int W_w, unsigned int C_in, unsigned int C_out){
    return c_out * (W_h * W_w * C_in) + h * (W_w * C_in) + w * C_in + c_in;
}

inline unsigned int idx_C(unsigned int h, unsigned int w, unsigned int c,
                    unsigned int C_h, unsigned int C_w, unsigned int C_c){
    return h * (C_w * C_c) + w * C_c + c;
}


void sp_conv_vertical_fp16_fp32_compute_ref(float16_t *C, float16_t const *A, float16_t const *B,
                  int A_h, int A_w, int W_h, int W_w, int C_in, unsigned int C_out,
                  int nitems) {
      
  unsigned int H_out = A_h - W_h + 1;
  unsigned int W_out = A_w - W_w + 1;
  unsigned int idxA = 0, idxB = 0, idxC = 0;
  float32_t acc = 0;
  for (unsigned int co = 0; co < C_out; co++) {
    for (unsigned int h = 0; h < H_out; h++) {
      for (unsigned int w = 0; w < W_out; w++) {
        acc = 0;
        for (unsigned short kh = 0; kh < W_h; kh++) {
          for (unsigned short kw = 0; kw < W_w; kw++) {
            for (unsigned short ci = 0; ci < C_in; ci++) {
              idxA = idx_A(h+kh, w+kw, ci, 
                                A_h, A_w, C_in);
              idxB = idx_B(kh, kw, ci, co, 
                                W_h, W_w, C_in, C_out);

              acc += (float32_t)A[idxA] * (float32_t)B[idxB];
            //   std::cout << acc << ": " << A[idxA] << ", " << B[idxB] << ", " << (float32_t)A[idxA] * (float32_t)B[idxB] << std::endl;
            }
          }
        }
        idxC = idx_C(h, w, co, 
                        H_out, W_out, C_out);
        C[idxC] = acc;
        // std::cout << acc << ": " << A[idxA] << ", " << B[idxB] << std::endl;
        // exit(0);
      }
    }
  }
}

void sp_conv_horizontal_fp16_fp32_compute_ref(float16_t *C, float16_t const *A, float16_t const *B,
                  int A_h, int A_w, int W_h, int W_w, int C_in, unsigned int C_out,
                  int nitems) {
      
  unsigned int H_out = A_h - W_h + 1;
  unsigned int W_out = A_w - W_w + 1;
  unsigned int idxA = 0, idxB = 0, idxC = 0;
  float32_t acc = 0;
  for (unsigned int co = 0; co < C_out; co++) {
    for (unsigned int h = 0; h < H_out; h++) {
      for (unsigned int w = 0; w < W_out; w++) {
        acc = 0;
        for (unsigned short kh = 0; kh < W_h; kh++) {
          for (unsigned short kw = 0; kw < W_w; kw++) {
            for (unsigned short ci = 0; ci < C_in; ci++) {
              idxA = idx_A(h+kh, w+kw, ci, 
                                A_h, A_w, C_in);
              idxB = idx_B(kh, kw, ci, co, 
                                W_h, W_w, C_in, C_out);

              acc += (float32_t)A[idxA] * (float32_t)B[idxB];
            //   std::cout << acc << ": " << A[idxA] << ", " << B[idxB] << ", " << (float32_t)A[idxA] * (float32_t)B[idxB] << std::endl;
            }
          }
        }
        idxC = idx_C(h, w, co, 
                        H_out, W_out, C_out);
        C[idxC] = acc;
        // std::cout << acc << ": " << A[idxA] << ", " << B[idxB] << std::endl;
        // exit(0);
      }
    }
  }
}

void dense_conv_ref(float16_t *C, float16_t const *A, float16_t const *B,
                  int A_h, int A_w, int W_h, int W_w, int C_in, unsigned int C_out,
                  int nitems) {
      
  unsigned int H_out = A_h - W_h + 1;
  unsigned int W_out = A_w - W_w + 1;
  unsigned int idxA = 0, idxB = 0, idxC = 0;
  float32_t acc = 0;
  for (unsigned int co = 0; co < C_out; co++) {
    for (unsigned int h = 0; h < H_out; h++) {
      for (unsigned int w = 0; w < W_out; w++) {
        acc = 0;
        for (unsigned short kh = 0; kh < W_h; kh++) {
          for (unsigned short kw = 0; kw < W_w; kw++) {
            for (unsigned short ci = 0; ci < C_in; ci++) {
              idxA = idx_A(h+kh, w+kw, ci, 
                                A_h, A_w, C_in);
              idxB = idx_B(kh, kw, ci, co, 
                                W_h, W_w, C_in, C_out);

              acc += (float32_t)A[idxA] * (float32_t)B[idxB];
            //   std::cout << acc << ": " << A[idxA] << ", " << B[idxB] << ", " << (float32_t)A[idxA] * (float32_t)B[idxB] << std::endl;
            }
          }
        }
        idxC = idx_C(h, w, co, 
                        H_out, W_out, C_out);
        C[idxC] = acc;
        // std::cout << acc << ": " << A[idxA] << ", " << B[idxB] << std::endl;
        // exit(0);
      }
    }
  }
}



#include <cassert>
#include <iostream>

#include "mm.h"

void sp_conv_vertical_fp16_fp32_compute_test(
                  float16_t *C, float16_t const *A_padded,
                  unsigned short const *W_join,
                  const unsigned int *W_indptr,
                  int A_h, int A_w, int W_h, int W_w, int C_in,
                  unsigned int C_out,
                  int nitems) {
  assert(nitems == 8);
  assert(8 == 8 or 8 == 16 or 8 == 12);
  unsigned int H_out = A_h - W_h + 1;
  unsigned int W_out = A_w - W_w + 1;
  const svbool_t pred16lo = svdupq_b16(false, false, false, false,
                                       true, true, true, true);
  const svbool_t pred16hi = svdupq_b16(true, true, true, true,
                                       false, false, false, false);
  const svbool_t pred16 = svptrue_b16();
  const svbool_t pred32 = svptrue_b32();
  for (unsigned int co = 0; co < C_out/nitems; co++) {
    unsigned int start = W_indptr[co];
    unsigned int end = W_indptr[co+1];
    //std::cout << start << " : " << end << std::endl;
    unsigned idx = start;
    unsigned step = 2 * nitems;
    while (idx + 8 <= end ) {
      svuint16_t W0_k = svld1_vnum_u16(pred16, &W_join[idx * step], 1);
      svuint16_t W0_v_u = svld1_vnum_u16(pred16, &W_join[idx * step], 0);
      svfloat16_t W0_v = svreinterpret_f16(W0_v_u);
      const svfloat32_t W0_v_hi = svcvt_f32_x(pred16hi, W0_v);
      const svfloat32_t W0_v_lo = svcvt_f32_x(pred16lo, W0_v);
      svuint16_t W1_k = svld1_vnum_u16(pred16, &W_join[idx * step], 3);
      svuint16_t W1_v_u = svld1_vnum_u16(pred16, &W_join[idx * step], 2);
      svfloat16_t W1_v = svreinterpret_f16(W1_v_u);
      const svfloat32_t W1_v_hi = svcvt_f32_x(pred16hi, W1_v);
      const svfloat32_t W1_v_lo = svcvt_f32_x(pred16lo, W1_v);
      svuint16_t W2_k = svld1_vnum_u16(pred16, &W_join[idx * step], 5);
      svuint16_t W2_v_u = svld1_vnum_u16(pred16, &W_join[idx * step], 4);
      svfloat16_t W2_v = svreinterpret_f16(W2_v_u);
      const svfloat32_t W2_v_hi = svcvt_f32_x(pred16hi, W2_v);
      const svfloat32_t W2_v_lo = svcvt_f32_x(pred16lo, W2_v);
      svuint16_t W3_k = svld1_vnum_u16(pred16, &W_join[idx * step], 7);
      svuint16_t W3_v_u = svld1_vnum_u16(pred16, &W_join[idx * step], 6);
      svfloat16_t W3_v = svreinterpret_f16(W3_v_u);
      const svfloat32_t W3_v_hi = svcvt_f32_x(pred16hi, W3_v);
      const svfloat32_t W3_v_lo = svcvt_f32_x(pred16lo, W3_v);
      svuint16_t W4_k = svld1_vnum_u16(pred16, &W_join[idx * step], 9);
      svuint16_t W4_v_u = svld1_vnum_u16(pred16, &W_join[idx * step], 8);
      svfloat16_t W4_v = svreinterpret_f16(W4_v_u);
      const svfloat32_t W4_v_hi = svcvt_f32_x(pred16hi, W4_v);
      const svfloat32_t W4_v_lo = svcvt_f32_x(pred16lo, W4_v);
      svuint16_t W5_k = svld1_vnum_u16(pred16, &W_join[idx * step], 11);
      svuint16_t W5_v_u = svld1_vnum_u16(pred16, &W_join[idx * step], 10);
      svfloat16_t W5_v = svreinterpret_f16(W5_v_u);
      const svfloat32_t W5_v_hi = svcvt_f32_x(pred16hi, W5_v);
      const svfloat32_t W5_v_lo = svcvt_f32_x(pred16lo, W5_v);
      svuint16_t W6_k = svld1_vnum_u16(pred16, &W_join[idx * step], 13);
      svuint16_t W6_v_u = svld1_vnum_u16(pred16, &W_join[idx * step], 12);
      svfloat16_t W6_v = svreinterpret_f16(W6_v_u);
      const svfloat32_t W6_v_hi = svcvt_f32_x(pred16hi, W6_v);
      const svfloat32_t W6_v_lo = svcvt_f32_x(pred16lo, W6_v);
      svuint16_t W7_k = svld1_vnum_u16(pred16, &W_join[idx * step], 15);
      svuint16_t W7_v_u = svld1_vnum_u16(pred16, &W_join[idx * step], 14);
      svfloat16_t W7_v = svreinterpret_f16(W7_v_u);
      const svfloat32_t W7_v_hi = svcvt_f32_x(pred16hi, W7_v);
      const svfloat32_t W7_v_lo = svcvt_f32_x(pred16lo, W7_v);

      for (unsigned int h = 0; h < H_out; h++) {
        for (unsigned int w = 0; w < W_out; w++) {
        //std::cout << idx << " : " << h << " : " << w << std::endl;
          svfloat32_t acc = svdup_f32(0);
          svfloat32_t acc1 = svdup_f32(0);
          float16_t const * A_addr = &A_padded[(h * A_w + w) * C_in];
          unsigned oidx = (h * W_out + w) * C_out + co*nitems; // NHWC
          const svbool_t pred64 = svptrue_b64();

          
          svfloat16_t A0_g = svreinterpret_f16(
            svld1_gather_index(pred64, (float64_t *)A_addr,
            svreinterpret_u64(W0_k)));
            std::cout <<" debug SVE load/gather instructions:" << std::endl;
            float16_t * W0_test = (float16_t *)calloc(8, sizeof(float16_t));
            uint16_t  * I0_test = (uint16_t *)calloc(8, sizeof(uint16_t));
            float16_t * A0_test = (float16_t *)calloc(8, sizeof(float16_t));
            svst1(pred16, &W0_test[0], W0_v);
            svst1(pred16, &I0_test[0], W0_k);
            svst1(pred16, &A0_test[0], A0_g);
            //std::cout <<" after the test" << std::endl;
            for(int m = 0; m < 8; m++){             
              std::cout << "weight: "<< W0_test[m] 
                        << ", index: "<< I0_test[m] 
                        << ", activation: " << A0_test[m] << std::endl;
            }
          const svfloat32_t A0_g_hi = svcvt_f32_x(pred16hi, A0_g);
          const svfloat32_t A0_g_lo = svcvt_f32_x(pred16lo, A0_g);
          acc = svmla_x(pred32, acc, A0_g_hi, W0_v_hi); //perform mla
          acc1 = svmla_x(pred32, acc1, A0_g_lo, W0_v_lo); //perform mla

          
          svfloat16_t A1_g = svreinterpret_f16(
            svld1_gather_index(pred64, (float64_t *)A_addr,
            svreinterpret_u64(W1_k)));
            std::cout <<" debug SVE load/gather instructions:" << std::endl;
            float16_t * W1_test = (float16_t *)calloc(8, sizeof(float16_t));
            uint16_t  * I1_test = (uint16_t *)calloc(8, sizeof(uint16_t));
            float16_t * A1_test = (float16_t *)calloc(8, sizeof(float16_t));
            svst1(pred16, &W1_test[0], W1_v);
            svst1(pred16, &I1_test[0], W1_k);
            svst1(pred16, &A1_test[0], A1_g);
            //std::cout <<" after the test" << std::endl;
            for(int m = 0; m < 8; m++){             
              std::cout << "weight: "<< W1_test[m] 
                        << ", index: "<< I1_test[m] 
                        << ", activation: " << A1_test[m] << std::endl;
            }
          const svfloat32_t A1_g_hi = svcvt_f32_x(pred16hi, A1_g);
          const svfloat32_t A1_g_lo = svcvt_f32_x(pred16lo, A1_g);
          acc = svmla_x(pred32, acc, A1_g_hi, W1_v_hi); //perform mla
          acc1 = svmla_x(pred32, acc1, A1_g_lo, W1_v_lo); //perform mla

          
          svfloat16_t A2_g = svreinterpret_f16(
            svld1_gather_index(pred64, (float64_t *)A_addr,
            svreinterpret_u64(W2_k)));
            std::cout <<" debug SVE load/gather instructions:" << std::endl;
            float16_t * W2_test = (float16_t *)calloc(8, sizeof(float16_t));
            uint16_t  * I2_test = (uint16_t *)calloc(8, sizeof(uint16_t));
            float16_t * A2_test = (float16_t *)calloc(8, sizeof(float16_t));
            svst1(pred16, &W2_test[0], W2_v);
            svst1(pred16, &I2_test[0], W2_k);
            svst1(pred16, &A2_test[0], A2_g);
            //std::cout <<" after the test" << std::endl;
            for(int m = 0; m < 8; m++){             
              std::cout << "weight: "<< W2_test[m] 
                        << ", index: "<< I2_test[m] 
                        << ", activation: " << A2_test[m] << std::endl;
            }
          const svfloat32_t A2_g_hi = svcvt_f32_x(pred16hi, A2_g);
          const svfloat32_t A2_g_lo = svcvt_f32_x(pred16lo, A2_g);
          acc = svmla_x(pred32, acc, A2_g_hi, W2_v_hi); //perform mla
          acc1 = svmla_x(pred32, acc1, A2_g_lo, W2_v_lo); //perform mla

          
          svfloat16_t A3_g = svreinterpret_f16(
            svld1_gather_index(pred64, (float64_t *)A_addr,
            svreinterpret_u64(W3_k)));
            std::cout <<" debug SVE load/gather instructions:" << std::endl;
            float16_t * W3_test = (float16_t *)calloc(8, sizeof(float16_t));
            uint16_t  * I3_test = (uint16_t *)calloc(8, sizeof(uint16_t));
            float16_t * A3_test = (float16_t *)calloc(8, sizeof(float16_t));
            svst1(pred16, &W3_test[0], W3_v);
            svst1(pred16, &I3_test[0], W3_k);
            svst1(pred16, &A3_test[0], A3_g);
            //std::cout <<" after the test" << std::endl;
            for(int m = 0; m < 8; m++){             
              std::cout << "weight: "<< W3_test[m] 
                        << ", index: "<< I3_test[m] 
                        << ", activation: " << A3_test[m] << std::endl;
            }
          const svfloat32_t A3_g_hi = svcvt_f32_x(pred16hi, A3_g);
          const svfloat32_t A3_g_lo = svcvt_f32_x(pred16lo, A3_g);
          acc = svmla_x(pred32, acc, A3_g_hi, W3_v_hi); //perform mla
          acc1 = svmla_x(pred32, acc1, A3_g_lo, W3_v_lo); //perform mla

          
          svfloat16_t A4_g = svreinterpret_f16(
            svld1_gather_index(pred64, (float64_t *)A_addr,
            svreinterpret_u64(W4_k)));
            std::cout <<" debug SVE load/gather instructions:" << std::endl;
            float16_t * W4_test = (float16_t *)calloc(8, sizeof(float16_t));
            uint16_t  * I4_test = (uint16_t *)calloc(8, sizeof(uint16_t));
            float16_t * A4_test = (float16_t *)calloc(8, sizeof(float16_t));
            svst1(pred16, &W4_test[0], W4_v);
            svst1(pred16, &I4_test[0], W4_k);
            svst1(pred16, &A4_test[0], A4_g);
            //std::cout <<" after the test" << std::endl;
            for(int m = 0; m < 8; m++){             
              std::cout << "weight: "<< W4_test[m] 
                        << ", index: "<< I4_test[m] 
                        << ", activation: " << A4_test[m] << std::endl;
            }
          const svfloat32_t A4_g_hi = svcvt_f32_x(pred16hi, A4_g);
          const svfloat32_t A4_g_lo = svcvt_f32_x(pred16lo, A4_g);
          acc = svmla_x(pred32, acc, A4_g_hi, W4_v_hi); //perform mla
          acc1 = svmla_x(pred32, acc1, A4_g_lo, W4_v_lo); //perform mla

          
          svfloat16_t A5_g = svreinterpret_f16(
            svld1_gather_index(pred64, (float64_t *)A_addr,
            svreinterpret_u64(W5_k)));
            std::cout <<" debug SVE load/gather instructions:" << std::endl;
            float16_t * W5_test = (float16_t *)calloc(8, sizeof(float16_t));
            uint16_t  * I5_test = (uint16_t *)calloc(8, sizeof(uint16_t));
            float16_t * A5_test = (float16_t *)calloc(8, sizeof(float16_t));
            svst1(pred16, &W5_test[0], W5_v);
            svst1(pred16, &I5_test[0], W5_k);
            svst1(pred16, &A5_test[0], A5_g);
            //std::cout <<" after the test" << std::endl;
            for(int m = 0; m < 8; m++){             
              std::cout << "weight: "<< W5_test[m] 
                        << ", index: "<< I5_test[m] 
                        << ", activation: " << A5_test[m] << std::endl;
            }
          const svfloat32_t A5_g_hi = svcvt_f32_x(pred16hi, A5_g);
          const svfloat32_t A5_g_lo = svcvt_f32_x(pred16lo, A5_g);
          acc = svmla_x(pred32, acc, A5_g_hi, W5_v_hi); //perform mla
          acc1 = svmla_x(pred32, acc1, A5_g_lo, W5_v_lo); //perform mla

          
          svfloat16_t A6_g = svreinterpret_f16(
            svld1_gather_index(pred64, (float64_t *)A_addr,
            svreinterpret_u64(W6_k)));
            std::cout <<" debug SVE load/gather instructions:" << std::endl;
            float16_t * W6_test = (float16_t *)calloc(8, sizeof(float16_t));
            uint16_t  * I6_test = (uint16_t *)calloc(8, sizeof(uint16_t));
            float16_t * A6_test = (float16_t *)calloc(8, sizeof(float16_t));
            svst1(pred16, &W6_test[0], W6_v);
            svst1(pred16, &I6_test[0], W6_k);
            svst1(pred16, &A6_test[0], A6_g);
            //std::cout <<" after the test" << std::endl;
            for(int m = 0; m < 8; m++){             
              std::cout << "weight: "<< W6_test[m] 
                        << ", index: "<< I6_test[m] 
                        << ", activation: " << A6_test[m] << std::endl;
            }
          const svfloat32_t A6_g_hi = svcvt_f32_x(pred16hi, A6_g);
          const svfloat32_t A6_g_lo = svcvt_f32_x(pred16lo, A6_g);
          acc = svmla_x(pred32, acc, A6_g_hi, W6_v_hi); //perform mla
          acc1 = svmla_x(pred32, acc1, A6_g_lo, W6_v_lo); //perform mla

          
          svfloat16_t A7_g = svreinterpret_f16(
            svld1_gather_index(pred64, (float64_t *)A_addr,
            svreinterpret_u64(W7_k)));
            std::cout <<" debug SVE load/gather instructions:" << std::endl;
            float16_t * W7_test = (float16_t *)calloc(8, sizeof(float16_t));
            uint16_t  * I7_test = (uint16_t *)calloc(8, sizeof(uint16_t));
            float16_t * A7_test = (float16_t *)calloc(8, sizeof(float16_t));
            svst1(pred16, &W7_test[0], W7_v);
            svst1(pred16, &I7_test[0], W7_k);
            svst1(pred16, &A7_test[0], A7_g);
            //std::cout <<" after the test" << std::endl;
            for(int m = 0; m < 8; m++){             
              std::cout << "weight: "<< W7_test[m] 
                        << ", index: "<< I7_test[m] 
                        << ", activation: " << A7_test[m] << std::endl;
            }
          const svfloat32_t A7_g_hi = svcvt_f32_x(pred16hi, A7_g);
          const svfloat32_t A7_g_lo = svcvt_f32_x(pred16lo, A7_g);
          acc = svmla_x(pred32, acc, A7_g_hi, W7_v_hi); //perform mla
          acc1 = svmla_x(pred32, acc1, A7_g_lo, W7_v_lo); //perform mla

          float32_t * acc_test = (float32_t *)calloc(4, sizeof(float32_t));
          float32_t * acc1_test = (float32_t *)calloc(4, sizeof(float32_t));
          svst1(pred16, &acc_test[0], acc);
          svst1(pred16, &acc1_test[0], acc1);
          std::cout <<" debug SVE cvt z/m instructions:" << std::endl;
          std::cout << " acc_test: ";  
          for(int m = 0; m < 4; m++){
              std::cout << acc_test[m] << " ";  
          }
          std::cout << "  acc1_test: ";  
          for(int m = 0; m < 4; m++){
              std::cout << acc1_test[m] << " ";  
          }
          std::cout << std::endl;  
          float16_t * acc16_test = (float16_t *)calloc(8, sizeof(float16_t));
          svfloat16_t acc16 = svcvt_f16_z(pred16lo, acc1);
          svst1(pred16, &acc16_test[0], acc16);
          std::cout << "  acc16_test: ";  
          for(int m = 0; m < 8; m++){
              std::cout << acc16_test[m] << " ";  
          }
          std::cout << std::endl; 
          acc16 = svcvt_f16_m(acc16, pred16hi, acc);
          svst1(pred16, &acc16_test[0], acc16);
          svst1(pred16, &C[oidx], acc16);
          std::cout << "  Result C matrix: ";  
          for(int m = 0; m < 8; m++){         
              std::cout << C[oidx+ m] << " ";
          }
          std::cout << std::endl; 
             exit(0);
        }
      }    
      idx = idx + 8;
    }
    while (idx + 4 <= end ) {
      svuint16_t W0_k = svld1_vnum_u16(pred16, &W_join[idx * step], 1);
      svuint16_t W0_v_u = svld1_vnum_u16(pred16, &W_join[idx * step], 0);
      svfloat16_t W0_v = svreinterpret_f16(W0_v_u);
      const svfloat32_t W0_v_hi = svcvt_f32_x(pred16hi, W0_v);
      const svfloat32_t W0_v_lo = svcvt_f32_x(pred16lo, W0_v);
      svuint16_t W1_k = svld1_vnum_u16(pred16, &W_join[idx * step], 3);
      svuint16_t W1_v_u = svld1_vnum_u16(pred16, &W_join[idx * step], 2);
      svfloat16_t W1_v = svreinterpret_f16(W1_v_u);
      const svfloat32_t W1_v_hi = svcvt_f32_x(pred16hi, W1_v);
      const svfloat32_t W1_v_lo = svcvt_f32_x(pred16lo, W1_v);
      svuint16_t W2_k = svld1_vnum_u16(pred16, &W_join[idx * step], 5);
      svuint16_t W2_v_u = svld1_vnum_u16(pred16, &W_join[idx * step], 4);
      svfloat16_t W2_v = svreinterpret_f16(W2_v_u);
      const svfloat32_t W2_v_hi = svcvt_f32_x(pred16hi, W2_v);
      const svfloat32_t W2_v_lo = svcvt_f32_x(pred16lo, W2_v);
      svuint16_t W3_k = svld1_vnum_u16(pred16, &W_join[idx * step], 7);
      svuint16_t W3_v_u = svld1_vnum_u16(pred16, &W_join[idx * step], 6);
      svfloat16_t W3_v = svreinterpret_f16(W3_v_u);
      const svfloat32_t W3_v_hi = svcvt_f32_x(pred16hi, W3_v);
      const svfloat32_t W3_v_lo = svcvt_f32_x(pred16lo, W3_v);

      for (unsigned int h = 0; h < H_out; h++) {
        for (unsigned int w = 0; w < W_out; w++) {
        //std::cout << idx << " : " << h << " : " << w << std::endl;
          svfloat32_t acc = svdup_f32(0);
          svfloat32_t acc1 = svdup_f32(0);
          float16_t const * A_addr = &A_padded[(h * A_w + w) * C_in];
          unsigned oidx = (h * W_out + w) * C_out + co*nitems; // NHWC
          const svbool_t pred64 = svptrue_b64();

          
          svfloat16_t A0_g = svreinterpret_f16(
            svld1_gather_index(pred64, (float64_t *)A_addr,
            svreinterpret_u64(W0_k)));
            std::cout <<" debug SVE load/gather instructions:" << std::endl;
            float16_t * W0_test = (float16_t *)calloc(8, sizeof(float16_t));
            uint16_t  * I0_test = (uint16_t *)calloc(8, sizeof(uint16_t));
            float16_t * A0_test = (float16_t *)calloc(8, sizeof(float16_t));
            svst1(pred16, &W0_test[0], W0_v);
            svst1(pred16, &I0_test[0], W0_k);
            svst1(pred16, &A0_test[0], A0_g);
            //std::cout <<" after the test" << std::endl;
            for(int m = 0; m < 8; m++){             
              std::cout << "weight: "<< W0_test[m] 
                        << ", index: "<< I0_test[m] 
                        << ", activation: " << A0_test[m] << std::endl;
            }
          const svfloat32_t A0_g_hi = svcvt_f32_x(pred16hi, A0_g);
          const svfloat32_t A0_g_lo = svcvt_f32_x(pred16lo, A0_g);
          acc = svmla_x(pred32, acc, A0_g_hi, W0_v_hi); //perform mla
          acc1 = svmla_x(pred32, acc1, A0_g_lo, W0_v_lo); //perform mla

          
          svfloat16_t A1_g = svreinterpret_f16(
            svld1_gather_index(pred64, (float64_t *)A_addr,
            svreinterpret_u64(W1_k)));
            std::cout <<" debug SVE load/gather instructions:" << std::endl;
            float16_t * W1_test = (float16_t *)calloc(8, sizeof(float16_t));
            uint16_t  * I1_test = (uint16_t *)calloc(8, sizeof(uint16_t));
            float16_t * A1_test = (float16_t *)calloc(8, sizeof(float16_t));
            svst1(pred16, &W1_test[0], W1_v);
            svst1(pred16, &I1_test[0], W1_k);
            svst1(pred16, &A1_test[0], A1_g);
            //std::cout <<" after the test" << std::endl;
            for(int m = 0; m < 8; m++){             
              std::cout << "weight: "<< W1_test[m] 
                        << ", index: "<< I1_test[m] 
                        << ", activation: " << A1_test[m] << std::endl;
            }
          const svfloat32_t A1_g_hi = svcvt_f32_x(pred16hi, A1_g);
          const svfloat32_t A1_g_lo = svcvt_f32_x(pred16lo, A1_g);
          acc = svmla_x(pred32, acc, A1_g_hi, W1_v_hi); //perform mla
          acc1 = svmla_x(pred32, acc1, A1_g_lo, W1_v_lo); //perform mla

          
          svfloat16_t A2_g = svreinterpret_f16(
            svld1_gather_index(pred64, (float64_t *)A_addr,
            svreinterpret_u64(W2_k)));
            std::cout <<" debug SVE load/gather instructions:" << std::endl;
            float16_t * W2_test = (float16_t *)calloc(8, sizeof(float16_t));
            uint16_t  * I2_test = (uint16_t *)calloc(8, sizeof(uint16_t));
            float16_t * A2_test = (float16_t *)calloc(8, sizeof(float16_t));
            svst1(pred16, &W2_test[0], W2_v);
            svst1(pred16, &I2_test[0], W2_k);
            svst1(pred16, &A2_test[0], A2_g);
            //std::cout <<" after the test" << std::endl;
            for(int m = 0; m < 8; m++){             
              std::cout << "weight: "<< W2_test[m] 
                        << ", index: "<< I2_test[m] 
                        << ", activation: " << A2_test[m] << std::endl;
            }
          const svfloat32_t A2_g_hi = svcvt_f32_x(pred16hi, A2_g);
          const svfloat32_t A2_g_lo = svcvt_f32_x(pred16lo, A2_g);
          acc = svmla_x(pred32, acc, A2_g_hi, W2_v_hi); //perform mla
          acc1 = svmla_x(pred32, acc1, A2_g_lo, W2_v_lo); //perform mla

          
          svfloat16_t A3_g = svreinterpret_f16(
            svld1_gather_index(pred64, (float64_t *)A_addr,
            svreinterpret_u64(W3_k)));
            std::cout <<" debug SVE load/gather instructions:" << std::endl;
            float16_t * W3_test = (float16_t *)calloc(8, sizeof(float16_t));
            uint16_t  * I3_test = (uint16_t *)calloc(8, sizeof(uint16_t));
            float16_t * A3_test = (float16_t *)calloc(8, sizeof(float16_t));
            svst1(pred16, &W3_test[0], W3_v);
            svst1(pred16, &I3_test[0], W3_k);
            svst1(pred16, &A3_test[0], A3_g);
            //std::cout <<" after the test" << std::endl;
            for(int m = 0; m < 8; m++){             
              std::cout << "weight: "<< W3_test[m] 
                        << ", index: "<< I3_test[m] 
                        << ", activation: " << A3_test[m] << std::endl;
            }
          const svfloat32_t A3_g_hi = svcvt_f32_x(pred16hi, A3_g);
          const svfloat32_t A3_g_lo = svcvt_f32_x(pred16lo, A3_g);
          acc = svmla_x(pred32, acc, A3_g_hi, W3_v_hi); //perform mla
          acc1 = svmla_x(pred32, acc1, A3_g_lo, W3_v_lo); //perform mla

          float32_t * acc_test = (float32_t *)calloc(4, sizeof(float32_t));
          float32_t * acc1_test = (float32_t *)calloc(4, sizeof(float32_t));
          svst1(pred16, &acc_test[0], acc);
          svst1(pred16, &acc1_test[0], acc1);
          std::cout <<" debug SVE cvt z/m instructions:" << std::endl;
          std::cout << " acc_test: ";  
          for(int m = 0; m < 4; m++){
              std::cout << acc_test[m] << " ";  
          }
          std::cout << "  acc1_test: ";  
          for(int m = 0; m < 4; m++){
              std::cout << acc1_test[m] << " ";  
          }
          std::cout << std::endl;  
          float16_t * acc16_test = (float16_t *)calloc(8, sizeof(float16_t));
          svfloat16_t acc16 = svcvt_f16_z(pred16lo, acc1);
          svst1(pred16, &acc16_test[0], acc16);
          std::cout << "  acc16_test: ";  
          for(int m = 0; m < 8; m++){
              std::cout << acc16_test[m] << " ";  
          }
          std::cout << std::endl; 
          acc16 = svcvt_f16_m(acc16, pred16hi, acc);
          svst1(pred16, &acc16_test[0], acc16);
          svst1(pred16, &C[oidx], acc16);
          std::cout << "  Result C matrix: ";  
          for(int m = 0; m < 8; m++){         
              std::cout << C[oidx+ m] << " ";
          }
          std::cout << std::endl; 
             exit(0);
        }
      }    
      idx = idx + 4;
    }
    while (idx + 2 <= end ) {
      svuint16_t W0_k = svld1_vnum_u16(pred16, &W_join[idx * step], 1);
      svuint16_t W0_v_u = svld1_vnum_u16(pred16, &W_join[idx * step], 0);
      svfloat16_t W0_v = svreinterpret_f16(W0_v_u);
      const svfloat32_t W0_v_hi = svcvt_f32_x(pred16hi, W0_v);
      const svfloat32_t W0_v_lo = svcvt_f32_x(pred16lo, W0_v);
      svuint16_t W1_k = svld1_vnum_u16(pred16, &W_join[idx * step], 3);
      svuint16_t W1_v_u = svld1_vnum_u16(pred16, &W_join[idx * step], 2);
      svfloat16_t W1_v = svreinterpret_f16(W1_v_u);
      const svfloat32_t W1_v_hi = svcvt_f32_x(pred16hi, W1_v);
      const svfloat32_t W1_v_lo = svcvt_f32_x(pred16lo, W1_v);

      for (unsigned int h = 0; h < H_out; h++) {
        for (unsigned int w = 0; w < W_out; w++) {
        //std::cout << idx << " : " << h << " : " << w << std::endl;
          svfloat32_t acc = svdup_f32(0);
          svfloat32_t acc1 = svdup_f32(0);
          float16_t const * A_addr = &A_padded[(h * A_w + w) * C_in];
          unsigned oidx = (h * W_out + w) * C_out + co*nitems; // NHWC
          const svbool_t pred64 = svptrue_b64();

          
          svfloat16_t A0_g = svreinterpret_f16(
            svld1_gather_index(pred64, (float64_t *)A_addr,
            svreinterpret_u64(W0_k)));
            std::cout <<" debug SVE load/gather instructions:" << std::endl;
            float16_t * W0_test = (float16_t *)calloc(8, sizeof(float16_t));
            uint16_t  * I0_test = (uint16_t *)calloc(8, sizeof(uint16_t));
            float16_t * A0_test = (float16_t *)calloc(8, sizeof(float16_t));
            svst1(pred16, &W0_test[0], W0_v);
            svst1(pred16, &I0_test[0], W0_k);
            svst1(pred16, &A0_test[0], A0_g);
            //std::cout <<" after the test" << std::endl;
            for(int m = 0; m < 8; m++){             
              std::cout << "weight: "<< W0_test[m] 
                        << ", index: "<< I0_test[m] 
                        << ", activation: " << A0_test[m] << std::endl;
            }
          const svfloat32_t A0_g_hi = svcvt_f32_x(pred16hi, A0_g);
          const svfloat32_t A0_g_lo = svcvt_f32_x(pred16lo, A0_g);
          acc = svmla_x(pred32, acc, A0_g_hi, W0_v_hi); //perform mla
          acc1 = svmla_x(pred32, acc1, A0_g_lo, W0_v_lo); //perform mla

          
          svfloat16_t A1_g = svreinterpret_f16(
            svld1_gather_index(pred64, (float64_t *)A_addr,
            svreinterpret_u64(W1_k)));
            std::cout <<" debug SVE load/gather instructions:" << std::endl;
            float16_t * W1_test = (float16_t *)calloc(8, sizeof(float16_t));
            uint16_t  * I1_test = (uint16_t *)calloc(8, sizeof(uint16_t));
            float16_t * A1_test = (float16_t *)calloc(8, sizeof(float16_t));
            svst1(pred16, &W1_test[0], W1_v);
            svst1(pred16, &I1_test[0], W1_k);
            svst1(pred16, &A1_test[0], A1_g);
            //std::cout <<" after the test" << std::endl;
            for(int m = 0; m < 8; m++){             
              std::cout << "weight: "<< W1_test[m] 
                        << ", index: "<< I1_test[m] 
                        << ", activation: " << A1_test[m] << std::endl;
            }
          const svfloat32_t A1_g_hi = svcvt_f32_x(pred16hi, A1_g);
          const svfloat32_t A1_g_lo = svcvt_f32_x(pred16lo, A1_g);
          acc = svmla_x(pred32, acc, A1_g_hi, W1_v_hi); //perform mla
          acc1 = svmla_x(pred32, acc1, A1_g_lo, W1_v_lo); //perform mla

          float32_t * acc_test = (float32_t *)calloc(4, sizeof(float32_t));
          float32_t * acc1_test = (float32_t *)calloc(4, sizeof(float32_t));
          svst1(pred16, &acc_test[0], acc);
          svst1(pred16, &acc1_test[0], acc1);
          std::cout <<" debug SVE cvt z/m instructions:" << std::endl;
          std::cout << " acc_test: ";  
          for(int m = 0; m < 4; m++){
              std::cout << acc_test[m] << " ";  
          }
          std::cout << "  acc1_test: ";  
          for(int m = 0; m < 4; m++){
              std::cout << acc1_test[m] << " ";  
          }
          std::cout << std::endl;  
          float16_t * acc16_test = (float16_t *)calloc(8, sizeof(float16_t));
          svfloat16_t acc16 = svcvt_f16_z(pred16lo, acc1);
          svst1(pred16, &acc16_test[0], acc16);
          std::cout << "  acc16_test: ";  
          for(int m = 0; m < 8; m++){
              std::cout << acc16_test[m] << " ";  
          }
          std::cout << std::endl; 
          acc16 = svcvt_f16_m(acc16, pred16hi, acc);
          svst1(pred16, &acc16_test[0], acc16);
          svst1(pred16, &C[oidx], acc16);
          std::cout << "  Result C matrix: ";  
          for(int m = 0; m < 8; m++){         
              std::cout << C[oidx+ m] << " ";
          }
          std::cout << std::endl; 
             exit(0);
        }
      }    
      idx = idx + 2;
    }
    while (idx + 1 <= end ) {
      svuint16_t W0_k = svld1_vnum_u16(pred16, &W_join[idx * step], 1);
      svuint16_t W0_v_u = svld1_vnum_u16(pred16, &W_join[idx * step], 0);
      svfloat16_t W0_v = svreinterpret_f16(W0_v_u);
      const svfloat32_t W0_v_hi = svcvt_f32_x(pred16hi, W0_v);
      const svfloat32_t W0_v_lo = svcvt_f32_x(pred16lo, W0_v);

      for (unsigned int h = 0; h < H_out; h++) {
        for (unsigned int w = 0; w < W_out; w++) {
        //std::cout << idx << " : " << h << " : " << w << std::endl;
          svfloat32_t acc = svdup_f32(0);
          svfloat32_t acc1 = svdup_f32(0);
          float16_t const * A_addr = &A_padded[(h * A_w + w) * C_in];
          unsigned oidx = (h * W_out + w) * C_out + co*nitems; // NHWC
          const svbool_t pred64 = svptrue_b64();

          
          svfloat16_t A0_g = svreinterpret_f16(
            svld1_gather_index(pred64, (float64_t *)A_addr,
            svreinterpret_u64(W0_k)));
            std::cout <<" debug SVE load/gather instructions:" << std::endl;
            float16_t * W0_test = (float16_t *)calloc(8, sizeof(float16_t));
            uint16_t  * I0_test = (uint16_t *)calloc(8, sizeof(uint16_t));
            float16_t * A0_test = (float16_t *)calloc(8, sizeof(float16_t));
            svst1(pred16, &W0_test[0], W0_v);
            svst1(pred16, &I0_test[0], W0_k);
            svst1(pred16, &A0_test[0], A0_g);
            //std::cout <<" after the test" << std::endl;
            for(int m = 0; m < 8; m++){             
              std::cout << "weight: "<< W0_test[m] 
                        << ", index: "<< I0_test[m] 
                        << ", activation: " << A0_test[m] << std::endl;
            }
          const svfloat32_t A0_g_hi = svcvt_f32_x(pred16hi, A0_g);
          const svfloat32_t A0_g_lo = svcvt_f32_x(pred16lo, A0_g);
          acc = svmla_x(pred32, acc, A0_g_hi, W0_v_hi); //perform mla
          acc1 = svmla_x(pred32, acc1, A0_g_lo, W0_v_lo); //perform mla

          float32_t * acc_test = (float32_t *)calloc(4, sizeof(float32_t));
          float32_t * acc1_test = (float32_t *)calloc(4, sizeof(float32_t));
          svst1(pred16, &acc_test[0], acc);
          svst1(pred16, &acc1_test[0], acc1);
          std::cout <<" debug SVE cvt z/m instructions:" << std::endl;
          std::cout << " acc_test: ";  
          for(int m = 0; m < 4; m++){
              std::cout << acc_test[m] << " ";  
          }
          std::cout << "  acc1_test: ";  
          for(int m = 0; m < 4; m++){
              std::cout << acc1_test[m] << " ";  
          }
          std::cout << std::endl;  
          float16_t * acc16_test = (float16_t *)calloc(8, sizeof(float16_t));
          svfloat16_t acc16 = svcvt_f16_z(pred16lo, acc1);
          svst1(pred16, &acc16_test[0], acc16);
          std::cout << "  acc16_test: ";  
          for(int m = 0; m < 8; m++){
              std::cout << acc16_test[m] << " ";  
          }
          std::cout << std::endl; 
          acc16 = svcvt_f16_m(acc16, pred16hi, acc);
          svst1(pred16, &acc16_test[0], acc16);
          svst1(pred16, &C[oidx], acc16);
          std::cout << "  Result C matrix: ";  
          for(int m = 0; m < 8; m++){         
              std::cout << C[oidx+ m] << " ";
          }
          std::cout << std::endl; 
             exit(0);
        }
      }    
      idx = idx + 1;
    }

  }
}
