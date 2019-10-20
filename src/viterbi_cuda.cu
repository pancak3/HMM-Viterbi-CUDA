#include <stdio.h>
#include <assert.h>
#include <float.h>

__global__ void max_probability(int n_states, int n_observations,
                                double *transition_matrix,
                                double *emission_table,
                                double *prev_probs, double *curr_probs,
                                int *backpaths);

__device__ void max(double const *vals, int n, double *max, int *idx_max);

__host__ int *viterbi_cuda(int n_states, int n_observations,
                           int const *observations, int observations_length,
                           double const *init_probabilities,
                           double **transition_matrix,
                           double **emission_table) {
    // allocate buffers on host to store backpaths and most likely path
    int *backpaths = (int *) calloc(observations_length * n_states,
                                    sizeof *backpaths);
    int *optimal_path = (int *) malloc(observations_length *
                                       sizeof *optimal_path);
    assert(backpaths && prev_probs && curr_probs && optimal_path);
#ifdef DEBUG
    printf("[HOST BUFFERS ALLOCATED] backpaths, optimal path\n");
#endif // DEBUG

    // allocate buffers on device for prev/curr state probs and curr backpaths
    double *dev_prev = NULL, *dev_curr = NULL; int *dev_backpaths = NULL;
    cudaMalloc(&dev_prev, n_states * sizeof *dev_prev);
    cudaMalloc(&dev_curr, n_states * sizeof *dev_curr);
    cudaMalloc(&dev_backpaths, n_states * sizeof *dev_backpaths);
    assert(dev_prev && dev_curr && dev_backpaths);
#ifdef DEBUG
    printf("[DEVICE BUFFERS ALLOCATED] dev_prev, dev_curr, dev_backpaths\n");
#endif // DEBUG

    // allocate buffer on device to store observation for each iteration
    int *dev_obs = NULL;
    cudaMalloc(&dev_obs, sizeof *dev_obs);
    assert(dev_obs);

    // allocate buffers on device for HMM params and copy from host
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

    // calculate initial state probabilities and copy to device memory
    double *temp = (double *) malloc(n_states * sizeof *temp);
    assert(temp);
    for (int i = 0; i < n_states; i++)
        temp[i] = init_probabilities[i] + emission_table[i][observations[0]];
    cudaMemcpy(dev_prev, temp, n_states * sizeof *temp, cudaMemcpyHostToDevice);
#ifdef DEBUG
    printf("[INIT STATE PROBS COPIED TO DEVICE MEMORY]\n");
#endif

    // calculate state probabilities for subsequent observations (parallel)
    for (int i = 1; i < observations_length; i++) {
        cudaMemcpy(dev_obs, observations + i, sizeof *dev_obs,
                   cudaMemcpyHostToDevice);
        max_probability<<<n_states, 32>>>(n_states, n_observations, dev_trans,
                                          dev_emission, dev_prev, dev_obs,
                                          curr_probs, dev_backpaths);
        cudaMemcpy(backpaths + i * n_states, dev_backpaths,
                   n_states * sizeof *backpaths, cudaMemcpyDeviceToHost);
        // swap pointers to treat curr probs as prev for next iteration
        double *temp = dev_prev;
        dev_prev = dev_curr;
        dev_curr = temp;
    }

    // determine highest final state probability
    cudaMemcpy(temp, dev_prev, n_states * sizeof *temp, cudaMemcpyDeviceToHost);
    double max = -DBL_MAX;
    for (int i = 0; i < n_states; i++)
        if (temp[i] > max) {
            max = temp[i];
            optimal_path[n_states - 1] = i;
        }

    // follow backpaths to determine all states
    for (int i = n_states - 1; i > 0; i--)
        optimal_path[i - 1] = backpaths[i][optimal_path[i]];

#ifdef DEBUG
    printf(" === BACKPATHS ===\n");
    for (int i = 0; i < n_observations; i++) {
        for (int j = 0; j < n_states; j++) {
            printf("%2d ", backpaths[i * n_states + j]);
        }
        putchar('\n');
    }
#endif

    free(temp);
    free(backpaths);
    cudaFree(dev_prev);
    cudaFree(dev_curr);
    cudaFree(dev_backpaths);
    cudaFree(dev_trans);
    cudaFree(dev_emission);
    return optimal_path;
}

__global__ void max_probability(int *n_states, int *n_observations,
                                double *transition_matrix,
                                double *emission_table,
                                double *prev_probs, int *observation,
                                double *curr_probs, int *backpaths) {
    int const bidx = blockIdx.x;
    int const tidx = threadIdx.x;

    if (tidx == 0) {
        double p_max = -DBL_MAX, p_temp;

        for (int i = 0; i < *n_states; i++) {
            p_temp = prev_probs[i] + transition_matrix[i * *n_states + bidx] +
                     emission_table[bidx * *n_observations + *observation];
            if (p_temp > p_max) {
                p_max = p_temp;
                i_max = i;
            }
        }
        curr_probs[bidx] = p_max;
        backpaths[bidx] = i_max;
    }
}

__device__ void max(double const *vals, int n, double *max, int *idx_max) {
    *max = -DBL_MAX;
    for (int i = 0; i < n; i++)
        if (vals[i] > *max) {
            *max = vals[i];
            *idx_max = i;
        }
}