

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

#include <cassert>
#include <cstdlib>
#include <iostream>

#include "checkpoint.h"
#include "mm.h"
#include "utils.h"


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


#define LOCAL_RAM_ADDR 536870912  //0x20000000
#define TCM 1

int main(int argc, char **argv) {
    unsigned int vector_length = svcnth();//Obtaining SVE vector length number
    size_t tcm_length_byte = sizeof(float16_t) * 2*B_H *B_W ;

    //float16_t *a = (float16_t *)calloc(A_H * A_W, sizeof(float16_t));
    void *addr = mmap((void *)LOCAL_RAM_ADDR, tcm_length_byte,
                    PROT_NONE, MAP_ANONYMOUS | MAP_PRIVATE,
                    -1, LOCAL_RAM_ADDR);

    float16_t *a = (float16_t*) addr;
    for (unsigned int i = 0; i < A_H * A_W; i++){
      a[i] =i*2.0;                // Activation A matrix manual inialization
    }

    unsigned short *b_data = (unsigned short *)calloc(B_H * B_W * 2,
      sizeof(unsigned short));
    float16_t *b_data_ref = (float16_t *)calloc(B_H * B_W , sizeof(float16_t));
    float16_t tmp = 0.0;
    for (unsigned int i = 0; i < 2*B_H *B_W/(vector_length*2); i++) {
      for (unsigned int cnt = 0; cnt < vector_length; cnt ++){
          tmp = (float16_t)1*cnt;
          b_data[vector_length*2*i+cnt] =
            reinterpret_cast<unsigned short &>(tmp);
            // Weight value B matrix manual inialization
          b_data[vector_length*(2*i+1)+cnt] = cnt + 16;
          // Weight index B matrix manual inialization
      }
    }
    /*
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
        std::cout << reinterpret_cast<float16_t &>(b_data[m]) << " ";
        //print weight value
        else if ((m/8) % 2 == 1)
        std::cout << (b_data[m])<< " ";  // print weight index
        if (m % 8 == 7)
        std::cout << std::endl;
    }*/

    const svbool_t pred16 = svptrue_b16();
    const svbool_t pred32 = svptrue_b32();
    const svbool_t pred64 = svptrue_b64();
    const svbool_t pred16lo = svdupq_b16(true, true, true, true,
                                       false, false, false, false);
    const svbool_t pred16hi = svdupq_b16(false, false, false, false,
                                       true, true, true, true);
    const svbool_t pred16even = svdupq_b16(true, false, true, false,
                                           true, false, true, false);
    const svbool_t pred16odd = svdupq_b16(false, true, false, true,
                                          false, true, false, true);
                                          
    std::cout << " start to test SVE B matrix load instructions:" << std::endl;
    svuint16_t W_k = svld1_vnum_u16(pred16, &b_data[0], 1);
    svuint16_t W_v_u = svld1_vnum_u16(pred16, &b_data[0], 0);
    svfloat16_t W_v = svreinterpret_f16(W_v_u);
    uint16_t  * W_k_sve = (uint16_t *)calloc(8, sizeof(uint16_t));
    float16_t * W_v_sve = (float16_t *)calloc(8, sizeof(float16_t));
    svst1(pred16, &W_k_sve[0], W_k);
    svst1(pred16, &W_v_sve[0], W_v);
    for (int m = 0; m < 8; m++){
        std::cout << "weight: "<< W_v_sve[m]
                << ", index: "<< W_k_sve[m]
                << ", expected  weight: "
                << reinterpret_cast<float16_t &>(b_data[m])
                << ", expected index: "<< b_data[8+m] << std::endl;
        assert(W_v_sve[m]== reinterpret_cast<float16_t &>(b_data[m]));
        assert(W_k_sve[m]== b_data[8+m]);
    }
    std::cout << " SVE load instructions pass test." << std::endl << std::endl;

    std::cout << " start to test SVE gather inst:" << std::endl;
    svfloat16_t A_g = svreinterpret_f16(
                            svld1_gather_index(pred64, (float64_t *)a,
                            svreinterpret_u64(W_k)));
    float16_t * A_g_sve = (float16_t *)calloc(8, sizeof(float16_t));
    svst1(pred16, &A_g_sve[0], A_g);
    for (int m = 0; m < 8; m++){
        std::cout << "gathered value: "<< A_g_sve[m]
                  << ", expected gathered value: "<< a[W_k_sve[m]]<< std::endl;
        assert(A_g_sve[m] == a[W_k_sve[m]]);
    }
    std::cout << " SVE gather inst pass test." << std::endl << std::endl;

    std::cout << " start to test SVE 32-bit cvt inst: " << std::endl;
    const svfloat32_t A_g_even = svcvt_f32_z(pred16, A_g);
    const svfloat16_t A_g_1 = svtrn2(A_g, A_g);
    const svfloat32_t A_g_odd = svcvt_f32_z(pred16, A_g_1);
    float32_t * A_g_even_test = (float32_t *)calloc(4, sizeof(float32_t));
    float32_t * A_g_odd_test = (float32_t *)calloc(4, sizeof(float32_t));
    svst1(pred32, &A_g_even_test[0], A_g_even);
    svst1(pred32, &A_g_odd_test[0], A_g_odd);
    for(int m = 0; m < 4; m++){
        std::cout << "cvted even value: " << (float32_t)A_g_even_test[m]
                  << ", expected cvted even value: " << (float32_t)A_g_sve[2*m]
                  << ", cvted odd value: " << A_g_odd_test[m]
                  << ", expected cvted odd value: "
                  << (float32_t)A_g_sve[2*m+1] << std::endl;
        assert(A_g_even_test[m] == (float32_t)A_g_sve[2*m]);
        assert(A_g_odd_test[m] == (float32_t)A_g_sve[2*m+1]);
    }
    std::cout << " SVE 32-bit cvt inst pass test." << std::endl << std::endl;
   
    std::cout << "start to test SVE svmla_x." << std::endl;
    svfloat32_t acc = svdup_f32(0);
    svfloat32_t acc1 = svdup_f32(0);
    const svfloat32_t W_v_even = svcvt_f32_z(pred16, W_v);
    const svfloat16_t W_v_1 = svtrn2(W_v, W_v);
    const svfloat32_t W_v_odd = svcvt_f32_z(pred16, W_v_1);

    acc = svmla_x(pred32, acc, A_g_even, W_v_even); //perform mla
    acc1 = svmla_x(pred32, acc1, A_g_odd, W_v_odd); //perform mla
    float32_t * acc_test = (float32_t *)calloc(4, sizeof(float32_t));
    float32_t * acc1_test = (float32_t *)calloc(4, sizeof(float32_t));
    svst1(pred32, &acc_test[0], acc);
    svst1(pred32, &acc1_test[0], acc1);

    for(int m = 0; m < 4; m++){
        std::cout << "acc value: " << (float32_t)acc_test[m]
                  << ", expected acc value: " << A_g_sve[2*m] * W_v_sve[2*m]
                  << ", acc1 value: " << (float32_t)acc1_test[m]
                  << ", expected acc1 value: " << A_g_sve[2*m+1] * W_v_sve[2*m+1] << std::endl;
        assert((float32_t)acc_test[m] == (float32_t)A_g_sve[2*m] * W_v_sve[2*m]);
        assert((float32_t)acc1_test[m] == (float32_t)A_g_sve[2*m+1] * W_v_sve[2*m+1]);
    }
    std::cout << " SVE svmla_x pass test." << std::endl << std::endl;

    std::cout << " start to test SVE 16-bit cvt inst: " << std::endl;
    svfloat16_t acc16_odd = svcvt_f16_z(pred16, acc1);
    svfloat16_t acc16_even = svcvt_f16_z(pred16, acc);
    svfloat16_t acc16 = svtrn1(acc16_even, acc16_odd);
    float16_t * acc16_test = (float16_t *)calloc(8, sizeof(float16_t));
    svst1(pred16, &acc16_test[0], acc16);
    for(int m = 0; m < 8; m++){
        std::cout << "cvted acc16 value: " << acc16_test[m]
                  << ", expected acc16 value: " << A_g_sve[m] * W_v_sve[m]<< std::endl;
        assert(acc16_test[m] == A_g_sve[m] * W_v_sve[m]);
    }
    std::cout << " SVE 16-bit cvt inst pass test." << std::endl << std::endl;


}
