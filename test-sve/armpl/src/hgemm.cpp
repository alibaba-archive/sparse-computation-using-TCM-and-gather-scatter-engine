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
//  __extension__ _Float16
#define T_DATA __fp16//_Float16


// void hgemm_(const char *transa, const char *transb, const armpl_int_t *m,
// 	const armpl_int_t *n, const armpl_int_t *k, const __fp16 *alpha,
// 	const __fp16 *A, const armpl_int_t *lda, const __fp16 *B,
// 	const armpl_int_t *ldb, const __fp16 *beta, __fp16 *C,
// 	const armpl_int_t *ldc, ... );

inline void read_data(const char* fname, T_DATA* alpha, T_DATA* beta){
        FILE* pFile;
        float alpha_32, beta_32;
        pFile = fopen(fname,"r");
        printf("Reading data from %s\n", fname);
        fscanf (pFile, "%f %f", &alpha_32,&beta_32);
        printf("alpha_32 = %f\n", alpha_32);
        printf(" beta_32 = %f\n", beta_32);
        alpha[0] = (T_DATA)alpha_32;
        beta[0] = (T_DATA)beta_32;

        printf("alpha = %f\n", alpha[0]);
        printf(" beta = %f\n", beta[0]);
}

int main(void){

        int m = 1024;
        int n = 1;
        int k = 1024;


        T_DATA *A, *B, *C;
        T_DATA *alpha, *beta;
        armpl_int_t * lda, *ldb, *ldc;



        const char transa = 'N';
        const char transb = 'N';
        A = (T_DATA *)malloc(m*k*sizeof(T_DATA));
        B = (T_DATA *)malloc(n*k*sizeof(T_DATA));
        C = (T_DATA *)malloc(m*n*sizeof(T_DATA));
        alpha = (T_DATA *)malloc(sizeof(T_DATA));
        beta = (T_DATA *)malloc(sizeof(T_DATA));
        lda = (armpl_int_t *)malloc(sizeof(armpl_int_t));
        ldb = (armpl_int_t *)malloc(sizeof(armpl_int_t));
        ldc = (armpl_int_t *)malloc(sizeof(armpl_int_t));

        const char* fname = "data_init.txt";
        read_data(fname, alpha, beta);



        lda[0] = m;
        ldb[0] = k;
        ldc[0] = m;
        for (int i=0; i<m*k; i++){
                A[i] = i*0.05;
                // printf("A[%d]=%f\t",i,C[i]);
        }
        for (int i=0; i<n*k; i++){
                B[i] = i*0.05;
                // printf("A[%d]=%f\t",i,C[i]);
        }
        // std::cout<<"begin hgemm"<<std::endl;
        reset_stats();
        hgemm_(&transa, &transb,
                &m, &n, &k,
                alpha, A, lda,
                B, ldb, beta, C, ldc);
        dump_stats();
        // std::cout<<"end hgemm"<<std::endl;
        // std::cout<<"c values=\n";
        for (int i=0; i<m*n; i++){
                printf("C[%d]=%f\t",i,C[i]);
        }

        // std::cout<<std::endl;
}
