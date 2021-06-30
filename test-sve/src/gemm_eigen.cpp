

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
#include <iostream>
#include <Eigen/Core>
// #include <Eigen/Half.h>
#include <bench/BenchTimer.h>
#include "checkpoint.h"
using namespace std;
using namespace Eigen;

#ifndef SCALAR
// #define SCALAR std::complex<float>
#define SCALAR  int16_t//__fp16//int16_t
//HLI: Currently gem5 doesn't support fmov for float16,
//so the matrix initialization is not working
#endif

#ifndef SCALARA
#define SCALARA SCALAR
#endif

#ifndef SCALARB
#define SCALARB SCALAR
#endif

typedef SCALAR Scalar;
typedef NumTraits<Scalar>::Real RealScalar;
typedef Matrix<SCALARA,Dynamic,Dynamic> A;
typedef Matrix<SCALARB,Dynamic,Dynamic> B;
typedef Matrix<Scalar,Dynamic,Dynamic> C;
typedef Matrix<RealScalar,Dynamic,Dynamic> M;

template<typename A, typename B, typename C>
EIGEN_DONT_INLINE void gemm(const A& a, const B& b, C& c)
{
 c.noalias() = a * b;
}

int main(int argc, char ** argv)
{

  int m = 1;
  int n = 1024;
  int p = 1024;


  A a(m,p); //a.setRandom(); //setRandom();
  B b(p,n); //b.setRandom(); //setRandom();
  C c(m,n); //c.setZero();

  std::cout << "Matrix sizes = " << m << "x" << p << " * " << p << "x" << n << "\n";


  reset_stats();
    gemm(a,b,c);
  dump_stats();

  return 0;
}

