
#!/bin/bash

set -x

rm -rf ./asm
mkdir -p ./asm

rm -rf ./bin
mkdir -p ./bin

gcc='aarch64-linux-gnu-g++'
objdump='aarch64-linux-gnu-objdump'
######  Scalar Version ######
## only add
$gcc -O3 -fno-tree-vectorize -g -Wall -march=armv8.2-a --static -S -o ./asm/daxpy-scalar.S ./src/daxpy-scalar.cpp
$gcc -O3 -fno-tree-vectorize -g -Wall -march=armv8.2-a --static -o ./bin/daxpy-scalar ./src/daxpy-scalar.cpp

## more complicated daxpy versionlinux-gnu-g++ -O3 -fno-tree-vectorize -g -Wall -march=armv8.2-a --static -S -o ./asm/daxpy-sve-scalar.S ./src/daxpy-sve.cpp
$gcc -O3 -fno-tree-vectorize -g -Wall -march=armv8.2-a --static -o ./bin/daxpy-sve-scalar ./src/daxpy-sve.cpp



######  SVE Intrinsic Version ######
# for executable
$gcc -O3 -g -Wall -march=armv8.2-a+sve --static -o ./bin/daxpy-sve-intrinsic ./src/daxpy-sve-intrinsic.cpp
# for assembly
$gcc -O3 -g -Wall -march=armv8.2-a+sve --static -S -o ./asm/daxpy-sve-intrinsic.S ./src/daxpy-sve-intrinsic.cpp



######  Auto Vectorization Version ######
## daxpy
$gcc -O3 -g -Wall -march=armv8.2-a+sve --static -S -o ./asm/daxpy-sve.S ./src/daxpy-sve.cpp
$gcc -O3 -g -Wall -march=armv8.2-a+sve --static -o ./bin/daxpy-sve ./src/daxpy-sve.cpp

## daxpy+gather
$gcc -O3 -g -Wall -march=armv8.2-a+sve --static -S -o ./asm/daxpy-sve-gather.S ./src/daxpy-sve-gather.cpp
$gcc -O3 -g -Wall -march=armv8.2-a+sve --static -o ./bin/daxpy-sve-gather ./src/daxpy-sve-gather.cpp


## gemm, test for auto vectorization
$gcc -O3 -g -Wall -march=armv8.2-a+sve --static -o ./asm/gemm.o ./src/gemm.cpp
$gcc -O3 -g -Wall -march=armv8.2-a+sve --static -o ./bin/gemm ./src/gemm.cpp
$objdump -d ./asm/gemm.o > ./dis/gemm.dis
