# gfortran  -O3 -g -Wall -march=armv8.2-a+simd  -I${ARMPL_DIR}/include -L${ARMPL_DIR}/lib --static -larmpl -larmpl_int64 -lamath -lm -lc -o hgemm  hgemm.cpp
g++ hgemm.cpp -O3 -g -Wall -march=armv8.2-a+simd+fp16  ${ARMPL_DIR}/lib/libarmpl.a -static  -o hgemm -lgfortran #-### 2> log.txt
objdump -d hgemm > hgemm.dis
#-larmpl -larmpl_mp -larmpl_int64 -larmpl_ilp64 -larmpl_ilp64_mp

g++ sgemv.cpp -O3 -g -Wall -static -march=armv8.2-a+simd+fp16 ${ARMPL_DIR}/lib/libarmpl.a -o sgemv -lgfortran #-### 2> log.txt
objdump -d sgemv > sgemv.dis
