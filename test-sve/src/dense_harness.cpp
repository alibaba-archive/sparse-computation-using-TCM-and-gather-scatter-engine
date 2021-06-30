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


// 512MB - 512MB+32kB
#define LOCAL_RAM_ADDR 536870912

#ifndef DIM_M
#define DIM_M 1
#endif
#ifndef DIM_N
#define DIM_N 128
#endif
#ifndef DIM_K
#define DIM_K 128
#endif

#if defined(DENSE_FP16)
  #define DENSE_DATA_T float16_t
#else
  #define DENSE_DATA_T float32_t
#endif

#define TCM 1


#if defined(CONV)
  // Define convolution of size H = 14 W = 14 Cin = 256 Cout = 256
  #define CONV_H 8
  #define CONV_W 8
  #define CONV_CIN 128
  #define CONV_COUT 128
  #define CONV_KH 3
  #define CONV_KW 3

  #define A_H 1
  #define A_W (CONV_H * CONV_W * CONV_CIN)
  #define B_H CONV_COUT
  #define B_W (CONV_KH * CONV_KW * CONV_CIN)
  #define C_H 1
  #define C_W ((CONV_H-CONV_KH+1)* (CONV_W-CONV_KW+1) * CONV_COUT)

  #else

  #define B_H DIM_N
  #define B_W DIM_K
  #define C_H DIM_M
  #define C_W DIM_N

 #if defined(TRANSPOSED)
  #define A_H DIM_K
 #define A_W DIM_M
 #else
 #define A_H DIM_M
 #define A_W DIM_K
 #endif // TRANSPOSED
 #endif



int main(int argc, char **argv) {
  // allocate the local memory
  // int fd = open("/dev/mem", O_RDWR);
  // A hack to map virtual address to physical address
  // LOCAL_RAM_ADDR needs to appear twice
  // MAP_ANONYMOUS must be set
  // fd == -1
  // see mmapFunc in syscall_emul.hh
  //reset_stats();
  #if TCM
  void *addr = mmap((void *)LOCAL_RAM_ADDR, 32 * 1024,
                    PROT_NONE, MAP_ANONYMOUS | MAP_PRIVATE,
                    -1, LOCAL_RAM_ADDR);

  if (addr == MAP_FAILED) {
    std::cout << "Failed to allocate on TCM" << std::endl;
  }
  std::cout << addr << std::endl;
  #endif

#if defined(TRANSPOSED)
  unsigned int A_M = DIM_K;
  unsigned int A_K = DIM_M;
#else
  unsigned int A_M = DIM_M;
  unsigned int A_K = DIM_K;
#endif

  DenseMatrix<float16_t> *bm = new DenseMatrix<float16_t>(B_H, B_W, false);
  float16_t *b = bm->get_values();
  std::cout << "B matrix starting address : " << std::hex
      << (unsigned long long) b  << std::dec << std::endl;
  std::cout << "B matrix size : " << bm->get_size() << std::endl;

  unsigned int vector_length = 0;
  vector_length = svcnth();
  // exit(0);
  // reset_stats();

#if (DIM_M == 1)
  #if TCM
  DenseMatrix<DENSE_DATA_T> *am = new DenseMatrix<DENSE_DATA_T>(addr,
                                                          A_H, A_W, false);
  #else
  DenseMatrix<DENSE_DATA_T> *am = new DenseMatrix<DENSE_DATA_T>(A_M, A_K,
    false);
  #endif
  DENSE_DATA_T *a = am->get_values();
  DenseMatrix<DENSE_DATA_T> *cm = new DenseMatrix<DENSE_DATA_T>(C_H, C_W,
    false);
  DENSE_DATA_T *c = cm->get_values();
#else
  DenseMatrix<DENSE_DATA_T> *am = new DenseMatrix<DENSE_DATA_T>(A_M, A_K,
    false);
  float16_t *a = am->get_values();
  DenseMatrix<DENSE_DATA_T> *cm = new DenseMatrix<DENSE_DATA_T>(DIM_M, DIM_N,
    false);
  float16_t *c = cm->get_values();
#endif

reset_stats();
#if defined(SCALAR)
  mm(c, a, b, DIM_M, DIM_K, DIM_N);
#elif defined(HGEMM)
  hgemm_01(c, a, b, DIM_M, DIM_K, DIM_N);
#elif defined(HGEMM_UNROLLED)
  hgemm_unrolled(c, a, b, DIM_M, DIM_K, DIM_N);
#elif defined(SCALAR_TRANSPOSED)
  mm_transposed(c, a, b, DIM_M, DIM_K, DIM_N);
#elif defined(HGEMM_UNROLLED_SPLIT)
  hgemm_unrolled_split(c, a, b, DIM_M, DIM_K, DIM_N);
#elif defined(VECTOR)
  #if defined(SPLIT)
  spmv_sve_split(c, a, b, DIM_K, DIM_N);
  #else
  //spmv_sve(c, a, b, DIM_K, DIM_N);
  mm_sve_split_fp16(c, a, b, DIM_K, DIM_N);
  // spmv_sve_fp16(c, a, b, DIM_K, DIM_N);
  #endif
#elif defined(CONV)
    dense_conv_fp16(c, a, b,
                  CONV_H, CONV_W, CONV_KH, CONV_KW, CONV_CIN,
                  CONV_COUT,
                  vector_length);
#else
  assert(false);
#endif
  dump_stats();
  //PrintMat(DIM_M, DIM_N, c);
}
