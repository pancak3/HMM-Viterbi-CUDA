#include <stdio.h>
#include <assert.h>
#include <float.h>

__global__ void max_probability(int n_states, int n_observations,
                                double *transition_matrix,
                                double *emission_table,
                                double *prev_probs, double *curr_probs,
                                int *backpaths);

__global__ void mult2(int *in, int *out) {
    out[0] = 2 * in[0];
    in[0] = 3 * in[0];
    __syncthreads();

    if (blockIdx.x == 1 && threadIdx.x == 8) {
        out[0] = blockIdx.x;
        in[0] = -1;
    }
}

__device__ void max(double const *vals, int n, double *max, int *idx_max);

__host__ int *viterbi_cuda(int n_states, int n_observations,
                           int const *observations, int observations_length,
                           double const *init_probabilities,
                           double **transition_matrix,
                           double **emission_table) {
    // allocate memory to store back-paths
    int *backpaths = (int *) calloc(observations_length * n_states,
                                    sizeof *backpaths);
    assert(backpaths);
#ifdef DEBUG
    printf("[BACK-PATHS TABLE ALLOCATED]\n");
#endif // DEBUG

    // allocate memory to store probabilities for previous and current time
    double *prev_probs = (double *) malloc(n_states * sizeof *prev_probs);
    double *curr_probs = (double *) malloc(n_states * sizeof *curr_probs);
    assert(prev_probs && curr_probs);



    // allocate memory to store final path
    int *optimal_path = NULL;
    cudaMalloc(&optimal_path, observations_length * sizeof *optimal_path);
    assert(optimal_path);
#ifdef DEBUG
    printf("[PATH ARRAY ALLOCATED]\n");
#endif

    // calculate state probabilities for time 0
    for (int i = 0; i < n_states; i++)
        prev_probs[i] = init_probabilities[i] +
                        emission_table[i][observations[0]];
#ifdef DEBUG
    printf("[INIT PROBS CALCULATED]\n");
#endif

    double *dev_prev;
    cudaMalloc(&dev_prev, n_states * sizeof *dev_prev);
    cudaMemcpy(dev_prev, prev_probs, n_states * sizeof *dev_prev,
               cudaMemcpyHostToDevice);
    double *dev_curr;
    cudaMalloc(&dev_curr, n_states * sizeof *dev_curr);
    cudaMemcpy(dev_curr, dev_curr, n_states * sizeof *dev_curr,
               cudaMemcpyHostToDevice);

#ifdef DEBUG
    printf("[INIT PROBS COPIED TO DEVICE MEMORY]\n");
#endif

    double *dev_trans;
    cudaMalloc(&dev_trans, n_states * n_states * sizeof *dev_trans);
    for (int i = 0; i < n_states; i++)
        cudaMemcpy(dev_trans + i * n_states, transition_matrix[i],
                   n_states * sizeof *dev_trans, cudaMemcpyHostToDevice);
#ifdef DEBUG
    printf("[TRANSITION MATRIX COPIED TO DEVICE MEMORY]\n");
#endif

    double *dev_emission;
    cudaMalloc(&dev_emission, n_states * n_observations * sizeof *dev_emission);
    for (int i = 0; i < n_states; i++)
        cudaMemcpy(dev_emission + i * n_observations, emission_table[i],
                   n_observations * sizeof *dev_emission,
                   cudaMemcpyHostToDevice);
#ifdef DEBUG
    printf("[EMISSION TABLE COPIED TO DEVICE MEMORY]\n");
#endif

    int *dev_n_states, *dev_n_obs;
    cudaMalloc(&dev_n_states, sizeof *dev_n_states);
    cudaMalloc(&dev_n_obs, sizeof *dev_n_obs);
    cudaMemcpy(dev_n_states, &n_states, sizeof n_states,
               cudaMemcpyHostToDevice);
    cudaMemcpy(dev_n_obs, &n_observations, sizeof n_observations,
               cudaMemcpyHostToDevice);
#ifdef DEBUG
    printf("[N_STATES AND N_OBSERVATIONS COPIED TO DEVICE MEMORY]\n");
#endif
    int *dev_backpaths;
    cudaMalloc(&dev_backpaths, n_states * sizeof *dev_backpaths);


#ifdef DEBUG
    printf("[BACKTRACK COPIED TO DEVICE MEMORY]\n");
#endif

    // calculate state probabilities for subsequent observations (parallel)
    for (int i = 1; i < observations_length; i++) {
        max_probability << < n_states, 32 >> > (n_states, n_observations,
                dev_trans, dev_emission,
                dev_prev, curr_probs,
                dev_backpaths);
        cudaMemcpy(backpaths + i * n_states, dev_backpaths,
                   n_states * sizeof *backpaths, cudaMemcpyDeviceToHost);
        double *temp = dev_prev;
        dev_prev = dev_curr;
        dev_curr = temp;
//        cudaMemcpy(prev_probs, &dev_curr, n_states * sizeof prev_probs,
//                   cudaMemcpyDeviceToHost);
//        cudaMemcpy(dev_prev, &prev_probs, n_states * sizeof dev_prev,
//                   cudaMemcpyHostToDevice);

    }
//    int c = 77;
//    cudaMemcpy(dev_backpaths, &c, sizeof *dev_backpaths, cudaMemcpyHostToDevice);
//    mult2 << < n_states, 32 >> > (dev_backpaths, dev_backpaths + 1);
//    cudaMemcpy(backpaths, dev_backpaths+1, sizeof *backpaths,cudaMemcpyDeviceToHost);
    printf(" =========================== \n");
    for (int i = 0; i < n_observations; i++) {
        for (int j = 0; j < n_states; j++) {
            printf("%d ", backpaths[i * n_states + j]);
        }
        printf("\n");
    }
    int *a;
    cudaMalloc(&a, 2*sizeof *a);
    int *b;
    cudaMalloc(&b, 2*sizeof *b);
    int c[2] = {88,99};
    cudaMemcpy(a, c, 2*sizeof *c, cudaMemcpyHostToDevice);
    mult2<<<8000, 64>>>(a, b);
    cudaMemcpy(c, b, 2*sizeof *c, cudaMemcpyDeviceToHost);
    printf("%d\n", c[0]);
    return NULL;
}

__global__ void max_probability(int n_states, int n_observations,
                                double *transition_matrix,
                                double *emission_table,
                                double *prev_probs, double *curr_probs,
                                int *backpaths) {
    int const bidx = blockIdx.x;
    int const tidx = threadIdx.x;

    double p_max = -DBL_MAX;
    double p_temp;
    int idx = -1;
    if (1) {
//    if (tidx == 0) {
        for (int i = 0; i < n_states; i++) {
            p_temp = prev_probs[i] + transition_matrix[bidx * n_states + i] +
                     emission_table[bidx * n_observations + i];
            if (p_temp > p_max) {
                p_max = p_temp;
                idx = i;
            }
        }
        curr_probs[bidx] = p_max;
        backpaths[bidx] = idx;
        backpaths[1] = 999;
    }
    backpaths[1] = 999;
    __syncthreads();




//    __shared__ int n_s;
//    __shared__ int n_obs;
//    extern __shared__ double pre[];
//    extern __shared__ double t_matrix[];
//    extern __shared__ double e_table[];
//
//    if (tidx == 0) {
//        n_s = n_states;
//        n_obs = n_observations;
//        for (int i = 0; i < n_s; i++) {
//            pre[i] = prev_probs[i];
//            for (int j = 0; j < n_s; j++)
//                t_matrix[i * n_s + j] = transition_matrix[i][j];
//            for (int j = 0; j < n_obs; j++)
//                e_table[i * n_obs + j] = emission_table[i][j];
//        }
//    }
//    __syncthreads();
}

__device__ void max(double const *vals, int n, double *max, int *idx_max) {
    *max = -DBL_MAX;
    for (int i = 0; i < n; i++)
        if (vals[i] > *max) {
            *max = vals[i];
            *idx_max = i;
        }
}