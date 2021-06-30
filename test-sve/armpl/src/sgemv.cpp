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


#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "armpl.h"
#include "checkpoint.h"

// #include <iostream>
#define T_DATA float

void sgemv_(const char *trans, const armpl_int_t *m,
            const armpl_int_t *n, const float *alpha,
            const float *a, const armpl_int_t *lda,
            const float *x, const armpl_int_t *incx,
            const float *beta, float *y, const armpl_int_t *incy, ... );



int main(void){

        int m = 1024;
        int n = 1024;
        // int k = 1024;

        T_DATA *a, *x, *y;
        T_DATA *alpha, *beta;
        armpl_int_t * lda, *incx, *incy;

        const char transa = 'N';
        // const char transb = 'N';
        a = (T_DATA *)malloc(m*n*sizeof(T_DATA));
        x = (T_DATA *)malloc(n*sizeof(T_DATA));
        y = (T_DATA *)malloc(m*sizeof(T_DATA));
        alpha = (T_DATA *)malloc(sizeof(T_DATA));
        beta = (T_DATA *)malloc(sizeof(T_DATA));
        lda = (armpl_int_t *)malloc(sizeof(armpl_int_t));
        incx = (armpl_int_t *)malloc(sizeof(armpl_int_t));
        incy = (armpl_int_t *)malloc(sizeof(armpl_int_t));
        alpha[0] = 1;
        beta[0] = 0;
        lda[0] = m;
        incx[0] = 1;
        incy[0] = 1;

        // std::cout<<"begin hgemm"<<std::endl;
        reset_stats();
        sgemv_(&transa, &m, &n, alpha, a, lda, x, incx, beta, y, incy);
        dump_stats();
        // std::cout<<"end hgemm"<<std::endl;
        // std::cout<<"c values=\n";
        // for (int i=0; i<m*n; i++){
        // 	std::cout<<C[i]<<"\t";
        // }

        // std::cout<<std::endl;
}
