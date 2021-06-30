/*
 * Copyright (c) 2020 Haoran Li
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

#include <arm_sve.h>

#include <cstdio>
#include <cstdlib>

//#pragma GCC aarch64 "arm_sve.h"

// Scalar version.
// void add_arrays(double * restrict dst, double *src, double c, const int N) {
//     for (int i = 0; i < N; i++)
//         dst[i] = src[i] + c;
// }

// Vector version
// void vla_add_arrays(double * restrict dst, double *src, double c, const int N) {
//     int64_t i = 0;

//     svbool_t pg = svwhilelt_b64(i, (int64_t)N);
//     while (svptest_any(svptrue_b64(), pg)) {
//         svfloat64_t vsrc = svld1(pg, src + i);
//         svfloat64_t vdst = svadd_x(pg, vsrc, c);
//         svst1(pg, dst + i, vdst);

//         i += svcntd();
//         pg = svwhilelt_b64(i, (int64_t)N);
//     }
// }

// Vector version
void vla_add_arrays_2(double *dst, double *src, double c, const int N) {
    for (int i = 0; i < N; i += svcntd()) {
        svbool_t Pg = svwhilelt_b64(i, N);
        svfloat64_t vsrc = svld1(Pg, &src[i]);
        svfloat64_t vdst = svadd_x(Pg, vsrc, c);
        svst1(Pg, &dst[i], vdst);
    }
}

int main(void) {
    const int vsize = 10000;
    double src[vsize];
    double c;
    double dst_serial[vsize], dst_vla[vsize], dst_vla2[vsize];
    for (int i = 0; i < vsize; ++i) {
        src[i] = (double) i / ((double) i + 1);
    }

    c = src[rand() % vsize];

    // add_arrays(dst_serial, src, c, vsize);
    // vla_add_arrays(dst_vla, src, c, vsize);
    vla_add_arrays_2(dst_vla2, src, c, vsize);

    // for (int i = 0; i < 100; ++i) {
    //     printf("%f %f %f, %f, %f\n", dst_serial[i], dst_vla[i], dst_vla2[i], src[i], c);
    // }
    return 0;
}
