
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
 * Author: Haoran Li
 */
#include <cstdio>
#include <random>
#include "mm.h"
#define DIM_M 128
#define DIM_N 128
#define DIM_K 128

#define TYPE_INDEX int
#define TYPE_ELEM float
// #pragma GCC aarch64 "arm_sve.h"
//HL: test if auto vectorization is supported
void mm(TYPE_ELEM *C, TYPE_ELEM const *A, TYPE_ELEM const *B,
  const TYPE_INDEX M, const TYPE_INDEX K,
  const TYPE_INDEX N) {
  for (TYPE_INDEX i = 0; i < M; ++i) {
    for (TYPE_INDEX j = 0; j < N; ++j) {
      C[i * N + j] = 0;
      for (TYPE_INDEX k = 0; k < K; ++k) {
        C[i * N + j] += A[i * K + k] * B[k * N + j];
      }
    }
  }
}

float *init_array(TYPE_INDEX m, TYPE_INDEX n, bool random) {
  float *res = (TYPE_ELEM *)calloc(m * n, sizeof(TYPE_ELEM));
  if (random) {
    for (TYPE_INDEX i = 0; i < m * n; i++) {
      // between -2 and 2
      res[i] = (rand() % 1000 - 500) / 250;
    }
  }
  return res;
}

int main(){
  TYPE_ELEM *a = init_array(DIM_M, DIM_K, true);
  TYPE_ELEM *b = init_array(DIM_K, DIM_N, true);
  TYPE_ELEM *c = init_array(DIM_M, DIM_N, true);

  mm(c, a, b, DIM_M, DIM_K, DIM_N);
  PrintMat(DIM_M,DIM_N,c);
}
