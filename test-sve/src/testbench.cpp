

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
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <math.h>

#include <cassert>
#include <cstdlib>
#include <iostream>

#include "checkpoint.h"
#include "mm.h"
#include "utils.h"

#if defined(CONV)
  #define CONV_H 3
  #define CONV_W 3
  #define CONV_CIN 8
  #define CONV_COUT 8
  #define CONV_KH 2
  #define CONV_KW 2

  #define A_H 1
  #define A_W (CONV_H * CONV_W * CONV_CIN)
  #define B_H CONV_COUT
  #define B_W (CONV_KH * CONV_KW * CONV_CIN)
  #define C_H 1
  #define C_W ((CONV_H - CONV_KH + 1) * (CONV_W - CONV_KW + 1) * CONV_COUT)

#else

#define B_H DIM_N
#define B_W DIM_K
#define C_H DIM_M
#define C_W DIM_N

#endif
//#define B_H 1024
//#define B_W 1024
//#define C_H B_H
//#define C_W A_W

#define LOCAL_RAM_ADDR 536870912  //0x20000000

#define TCM 1
#if TCM
#define TCM_NUM_ELEM 32768
  #if defined(DENSE_FP16)
    #define TCM_NUM_BYTE_PER_ELEM 2
  #else
    #define TCM_NUM_BYTE_PER_ELEM 4
  #endif
// #define TCM_NUM_BYTE_PER_ELEM 32
#endif



int main(int argc, char **argv) {
    unsigned int vector_length = svcnth();       //Obtaining SVE vector length number
    size_t tcm_length_byte = sizeof(float16_t) * 2*B_H *B_W ;
    //float16_t *a = (float16_t *)calloc(A_H * A_W, sizeof(float16_t));
    void *addr = mmap((void *)LOCAL_RAM_ADDR, tcm_length_byte,
                    PROT_NONE, MAP_ANONYMOUS | MAP_PRIVATE,
                    -1, LOCAL_RAM_ADDR);


    #if defined(CONV)
        #if defined(SBUCKET)
            float16_t *a = (float16_t*) addr;
            unsigned short *b_data = (unsigned short *)calloc(B_H * B_W * 2, sizeof(unsigned short));
            float16_t *b_data_ref = (float16_t *)calloc(B_H * B_W , sizeof(float16_t));
            //float16_t simple_a[vector_length] = {1,2,3,4,5,6,7,8}; // simple test

            #if defined(V_MODE)
                for (unsigned int i = 0; i < A_H * A_W; i++){
                    a[i] =(float16_t)0.1*i;                              // Activation A matrix manual inialization 
                }
                
                float16_t tmp = 0.0;
                for (unsigned int i = 0; i < 2*B_H *B_W/(vector_length*2); i++) {
                    for (unsigned int cnt = 0; cnt < vector_length; cnt ++){
                        tmp = (float16_t)1*i;
                        b_data[vector_length*2*i+cnt] = reinterpret_cast<unsigned short &>(tmp);   // Weight value B matrix manual inialization  
                        // b_index need offset after second raw
                        b_data[vector_length*(2*i+1)+cnt] = i>=2*B_H *B_W/(vector_length*2)/2 ? i+8 : i;   // Weight index B matrix manual inialization  
                    }
                }
            
                for(unsigned int i = 0; i < B_H; i++){
                    for(unsigned int j = 0; j < B_W; j++){
                        b_data_ref[i*B_W + j] = (float16_t)1*j;      // Reference Weight value B matrix manual inialization
                    }
                }
            #elif defined(H_MODE)
                for (unsigned int i = 0; i < A_H * A_W; i++){
                    a[i] =(float16_t)1*i;                              // Activation A matrix manual inialization 
                }
                
                float16_t tmp = 0.0;
                for (unsigned int i = 0; i < 2*B_H *B_W/(vector_length*2); i++) {
                    for (unsigned int cnt = 0; cnt < vector_length; cnt ++){
                        tmp = (float16_t)1*cnt;
                        b_data[vector_length*2*i+cnt] = reinterpret_cast<unsigned short &>(tmp);   // Weight value B matrix manual inialization  
                        // b_index need offset after second raw
                        b_data[vector_length*(2*i+1)+cnt] = (i%4)*8 >=16 ? (i%4)*8+cnt+8 : (i%4)*8+cnt;   // Weight index B matrix manual inialization  
                    }
                }
            
                for(unsigned int i = 0; i < B_H; i++){
                    for(unsigned int j = 0; j < B_W/8; j++){
                        for(unsigned int k = 0; k < 8; k++){
                            b_data_ref[i*B_W + j*8 + k] = (float16_t)1 * k;      // Reference Weight value B matrix manual inialization
                        }
                    }
                }
            #endif
        #else //SBLOCK
            float16_t *a = (float16_t*) addr;
            float16_t *b_data = (float16_t *)calloc(B_H * B_W, sizeof(float16_t));
            unsigned short *b_indice = (unsigned short *)calloc(B_H * B_W / 8, sizeof(unsigned short));
            float16_t *b_data_ref = (float16_t *)calloc(B_H * B_W , sizeof(float16_t));
            //float16_t simple_a[vector_length] = {1,2,3,4,5,6,7,8}; // simple test

            #if defined(V_MODE)
                for (unsigned int i = 0; i < A_H * A_W; i++){
                    a[i] =(float16_t)0.1*i;                              // Activation A matrix manual inialization 
                }
                
                float16_t tmp = 0.0;
                for (unsigned int i = 0; i < B_H *B_W/vector_length; i++) {
                    for (unsigned int cnt = 0; cnt < vector_length; cnt ++){
                        b_data[vector_length*i+cnt] = (float16_t)1*i;   // Weight value B matrix manual inialization  
                    }
                    b_indice[i] = i>=B_H *B_W/vector_length/2 ? i+8 : i;   // Weight index B matrix manual inialization  
                }
            
                for(unsigned int i = 0; i < B_H; i++){
                    for(unsigned int j = 0; j < B_W; j++){
                        b_data_ref[i*B_W + j] = (float16_t)1*j;      // Reference Weight value B matrix manual inialization
                    }
                }
            #elif defined(H_MODE)
                for (unsigned int i = 0; i < A_H * A_W; i++){
                    a[i] =(float16_t)1*i;                              // Activation A matrix manual inialization 
                }
                
                float16_t tmp = 0.0;
                for (unsigned int i = 0; i < B_H *B_W/vector_length; i++) {
                    for (unsigned int cnt = 0; cnt < vector_length; cnt ++){
                        b_data[vector_length*i+cnt] = (float16_t)1*cnt;   // Weight value B matrix manual inialization 
                    }
                    b_indice[i] = (i%4)*8 >=16 ? (i%4)*1 + 1 : (i%4)*1;   // Weight index B matrix manual inialization  
                    
                }
            
                for(unsigned int i = 0; i < B_H; i++){
                    for(unsigned int j = 0; j < B_W/8; j++){
                        for(unsigned int k = 0; k < 8; k++){
                            b_data_ref[i*B_W + j*8 + k] = (float16_t)1 * k;      // Reference Weight value B matrix manual inialization
                        }
                    }
                }
            #endif
        #endif
    #endif

#if defined(SBUCKET)
  std::cout<<"Print the generated Matrix A:"<< std::endl;
  for (int m = 0; m<A_H * A_W;m ++)
   {
     std::cout<<a[m]<< ", ";
   }
  
  std::cout<< std::endl;
  std::cout<<"Print the generated Matrix B:"<< std::endl;
   for (int m = 0; m< 2*B_H*B_W; m ++)
  {
     if (m % 8 == 0 && (m/8) % 2 == 0)
      std::cout << "value: ";
     else if (m % 8 == 0 && (m/8) % 2 == 1)
      std::cout << "index: ";
     if ((m/8) % 2 == 0)
      std::cout << reinterpret_cast<float16_t &>(b_data[m]) << " ";  //print weight value
     else if ((m/8) % 2 == 1)
      std::cout << (b_data[m])<< " ";  // print weight index
     if (m % 8 == 7)
      std::cout << std::endl;
  }
#else
  std::cout<<"Print the generated Matrix A:"<< std::endl;
  for (int m = 0; m < A_H * A_W;m ++)
   {
     std::cout<<a[m]<< ", ";
   }
  
  std::cout<< std::endl;
  std::cout<<"Print the generated Matrix B:"<< std::endl;
   for (int m = 0; m< B_H*B_W; m ++)
  {
     if (m % 8 == 0){
       std::cout << "index: ";
       std::cout << b_indice[m/8]<< " ";  // print weight index
       std::cout << "value: ";
     }
     std::cout << b_data[m] << " ";  //print weight value
     if (m % 8 == 7)
      std::cout << std::endl;
  }
#endif



#if defined(CONV) 
  float16_t *c = (float16_t *)calloc(C_H * C_W, sizeof(float16_t));
  float16_t *c_correct = (float16_t *)calloc(C_H * C_W, sizeof(float16_t));
#endif

#if defined(V_MODE)   //test vertical kernels
    unsigned int *b_indptr = (unsigned int *)calloc(B_H / vector_length + 1, sizeof(unsigned int));
    unsigned int simple_b_indptr[B_H/vector_length +1] = {0, B_W}; // use simple test
    for (unsigned int i = 0; i < B_H / vector_length +1; i++){
      b_indptr[i] = simple_b_indptr[i];
    }
#elif defined(H_MODE)  // test horizontal kernels
    unsigned int *b_indptr = (unsigned int *)calloc(B_H + 1, sizeof(unsigned int));
    //unsigned int simple_b_indptr[B_W +1] = {0,1,2,3,4,5,6,7,8};
    for (unsigned int i = 0; i < B_H +1; i++){
      //b_indptr[i] = simple_b_indptr[i];
        b_indptr[i] = 4*i;
    }
#endif
  

#if defined(DENSE_FP16)    
    dense_conv_fp16(c, a, b_data_ref,
                  CONV_H, CONV_W, CONV_KH, CONV_KW, CONV_CIN,
                  CONV_COUT,
                  vector_length);
    sp_conv_vertical_fp16_fp32_compute_ref(
                        c_correct, a, b_data_ref, CONV_H, CONV_W,
                        CONV_KH, CONV_KW, CONV_CIN, CONV_COUT,
                        vector_length);   // reference target kernel
    for (unsigned int i = 0; i < C_H * C_W; i++){
        std::cout << "ver result in C" << i << "  ";
        std::cout << "kernel: " << c[i] << "  ";
        std::cout << "correct: " << c_correct[i] << std::endl;
        // assert(fabs(c[i] - c_correct[i]) < 0.01);
    }
#else
    #if defined(V_MODE)
    #if defined(CONV)    // test CONV kernels
        #if defined(SBUCKET)
            std::cout << "sp_conv_vertical_fp16_fp32_compute" << std::endl;
            sp_conv_vertical_fp16_fp32_compute(
                            c, a, b_data, b_indptr, CONV_H, CONV_W,
                            CONV_KH, CONV_KW, CONV_CIN, CONV_COUT,
                            vector_length);    // test target kernel
            sp_conv_vertical_fp16_fp32_compute_ref(
                            c_correct, a, b_data_ref, CONV_H, CONV_W,
                            CONV_KH, CONV_KW, CONV_CIN, CONV_COUT,
                            vector_length);   // reference target kernel
        #else
            std::cout << "sp_conv_block_vertical_fp16_fp32_compute" << std::endl;
            sp_conv_block_vertical_fp16_fp32_compute(
                    c, a, b_data, b_indice,b_indptr, CONV_H, CONV_W,
                            CONV_KH, CONV_KW, CONV_CIN, CONV_COUT,
                            vector_length);
            sp_conv_vertical_fp16_fp32_compute_ref(
                            c_correct, a, b_data_ref, CONV_H, CONV_W,
                            CONV_KH, CONV_KW, CONV_CIN, CONV_COUT,
                            vector_length);   // reference target kernel
        #endif
    #else          // test SPMV kernels
    spmv_joined_bucket_vertical_densefp16(c, a, b_data, {b_indptr},
                        A_W, A_H, vector_length);

    spmv_vertical_correct(c_correct, a, b_data, b_indptr,
                        A_W, A_H, vector_length);
    #endif                    

    for (unsigned int i = 0; i < C_H * C_W; i++){
        std::cout << "ver result in C" << i << "  ";
        std::cout << "kernel: " << c[i] << "  ";
        std::cout << "correct: " << c_correct[i] << std::endl;
        assert(fabs(c[i] - c_correct[i]) < 0.01);
    }
    
    #elif defined(H_MODE)
    #if defined(CONV)    // test CONV kernels
        #if defined(SBUCKET)
            std::cout << "sp_conv_horizontal_fp16_fp32_compute" << std::endl;
            sp_conv_horizontal_fp16_fp32_compute(
                            c, a, b_data, b_indptr, CONV_H, CONV_W,
                            CONV_KH, CONV_KW, CONV_CIN, CONV_COUT,
                            vector_length);    // test target kernel
            sp_conv_horizontal_fp16_fp32_compute_ref(
                            c_correct, a, b_data_ref, CONV_H, CONV_W,
                            CONV_KH, CONV_KW, CONV_CIN, CONV_COUT,
                            vector_length);   // reference target kernel
        #else
            std::cout << "sp_conv_block_horizontal_fp16_fp32_compute" << std::endl;
            sp_conv_block_horizontal_fp16_fp32_compute(
                    c, a, b_data, b_indice,b_indptr, CONV_H, CONV_W,
                            CONV_KH, CONV_KW, CONV_CIN, CONV_COUT,
                            vector_length);
            sp_conv_horizontal_fp16_fp32_compute_ref(
                            c_correct, a, b_data_ref, CONV_H, CONV_W,
                            CONV_KH, CONV_KW, CONV_CIN, CONV_COUT,
                            vector_length);   // reference target kernel

        #endif
    #else          // test SPMV kernels

        spmv_joined_bucket_horizontal_fp16(c, a, b_data, b_indptr,
                            A_W, A_H, vector_length);
        spmv_horizon_correct(c_correct, a, b_data, b_indptr,
                            A_W, A_H, vector_length);

    #endif                    

    for (unsigned int i = 0; i < C_H * C_W; i++){
        std::cout << "hor result in C" << i << "  ";
        std::cout << "kernel: " << c[i] << "  ";
        std::cout << "correct: " << c_correct[i] << std::endl;
        assert(fabs(c[i] - c_correct[i]) < 0.01);
    }

    #else

    #endif
#endif
  
  
}
