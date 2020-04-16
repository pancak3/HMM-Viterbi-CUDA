driver_cuda:
	gcc util/generator.c -o bin/generator
	nvcc -rdc=true -arch=sm_50 src/driver.cu src/viterbi_sequential.cu src/viterbi_cuda.cu -o bin/driver_cuda
debug:
	gcc util/generator.c -o bin/generator
	nvcc -D DEBUG -rdc=true -arch=sm_50 src/driver.cu src/viterbi_sequential.cu src/viterbi_cuda.cu -o bin/driver_cuda
clean:
	rm -rf $(EXE) *.o obj
format:
	find . -name "*.c" -or -name "*.cu"  -or -name "*.h"| xargs clang-format -style=file -i -fallback-style=none

