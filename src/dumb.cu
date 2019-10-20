#include <stdio.h>

__global__ void mult2(int *in, int *out) {
    out[0] = 2 * in[0];
    in[0] = 3 * in[0];
    __syncthreads();

    if (blockIdx.x == 7999 && threadIdx.x == 8) {
        out[0] = blockIdx.x;
        in[0] = -1;
    }
}

int main(void) {
    int *a;
    cudaMalloc(&a, 2*sizeof *a);
    int *b;
    cudaMalloc(&b, 2*sizeof *b);
    int c[2] = {88,99};
    cudaMemcpy(a, c, 2*sizeof *c, cudaMemcpyHostToDevice);
    mult2<<<8000, 64>>>(a, b);
    cudaMemcpy(c, b, 2*sizeof *c, cudaMemcpyDeviceToHost);
    printf("%d\n", c[0]);
    return 0;
}