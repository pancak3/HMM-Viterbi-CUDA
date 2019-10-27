# Viterbi_CUDA
We implement the Viterbi Algorithm both in sequential and in parallel.
The sequential one is `src/viterbi_sequential.cu` and the parallel one using
  CUDA which is `src/viterbi_cuda.cu`. <br>

To learn more details about Viterbi Algorithm, please refer to our report. We
 only introduce the dependencies and running environments here.<br>
<strong>NOTE:</strong><br>
Our algorithm were tested on `Ubuntu 18.04` with `Nvidia Pascal P100`, due to the
 difference of  the architecture of Nvidia products, you may define `THREADS_PER_BLOCK` at `line
  5` of  `src/viterbi_cuda.cu` according to your Nvidia card.
## Dependency
-   [CUDA 10.1](https://developer.nvidia.com/cuda-10.1-download-archive
)
-   GCC ~ 7.4.0
 
## Environment
+ PATH=$PATH:/usr/local/cuda-10.1/bin
+ LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64

## Instructions
To check CUDA environment,
```shell script
user@machie$ nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2019 NVIDIA Corporation
Built on Sun_Jul_28_19:07:16_PDT_2019
Cuda compilation tools, release 10.1, V10.1.243
```
To compile all the files without debug info (use `build_debug.sh` for debug
 info),
```shell script
user@machie$ ./build.sh 
[*] use nvcc to compile 
	{driver.cu, viterbi_cuda.cu,viterbi_sequential.cu}
	... done.
	--> bin/driver_cuda
[*] use gcc to compile ... done.
	--> bin/generator
```
To generate problem with `1000` states, `2000` possible observations and `100`
 actual observations,
```shell script
user@machie$ ./bin/generator > problem.in
1000 2000 100
user@machie$ 
```
To compare Viterbi_Sequential and Viterbi_CUDA,
```shell script
user@machie$ ./bin/driver_cuda < problem.in 
States: 1000, Emissions: 2000, Observation_length: 100
SEQUENTIAL Time: 1081669
CUDA Time: 37556
[ SEQUENTIAL OPTIMAL PATH ]
  0   2   0   1   3   0   2   1   0   4   2   0   0   0   4   3   0   1   3   0   1   3   0   0   1   3   0   1   3   0   1   3   0   1   3   0   1   3   0   4   3   0   0   0   1   3   0   4   3   0   0   1   3   0   0   2   4   2   1   3   0   1   3   0   4   3   0   4   2   0   0   0   0   4   3   0   0   0   0   1   3   0   1   3   0   0   0   1   3   0   1   3   0   9   2   1   3   0   0   1 
[ CUDA OPTIMAL PATH ]
  0   2   0   1   3   0   2   1   0   4   2   0   0   0   4   3   0   1   3   0   1   3   0   0   1   3   0   1   3   0   1   3   0   1   3   0   1   3   0   4   3   0   0   0   1   3   0   4   3   0   0   1   3   0   0   2   4   2   1   3   0   1   3   0   4   3   0   4   2   0   0   0   0   4   3   0   0   0   0   1   3   0   1   3   0   0   0   1   3   0   1   3   0   9   2   1   3   0   0   1
user@machie$ 
```


