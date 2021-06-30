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

#include <cstdio>
#include <random>
#include <iostream>
using namespace std;
#pragma GCC aarch64 "arm_sve.h"
int main()
    {
        const int N = 10000;
        double X[N],Z[N],K[N], alpha = 0.45;
        int Y[N];
        std::random_device rd; std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0, N-1);

        for (int i = 0; i < N; ++i)
        {
            X[i] = i;
            Y[i] = (int)dis(gen);
            K[i] = dis(gen);
        }
        cout<<"finished vector initialization\n";
        // Start of daxpy loop
        for (int i = 0; i < N; ++i)
        {
            Z[i] = alpha * X[Y[i]] + K[i];
            // cout<<"i="<<i<<"\tX[Y[i]]="<<X[Y[i]]<<"\tY[i]"<<Y[i]<<"\tK[i]="<<K[i]<<endl; 

        }
        // End of daxpy loop
        cout<<"finished daxpy loop\n";
        //reduction
        double sum = 0;
        for (int i = 0; i < N; ++i)
        {
            sum += Z[i];
        }
        cout<<"finished reduction, sum = "<<sum<<endl;
        return 0; 
    }
