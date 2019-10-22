#include <stdio.h>
#include <assert.h>
#include <float.h>

#define THREADS_PER_BLOCK 32

__global__ void max_probability(int *n_states,
                                int *n_possible_observations,
                                double *transition_matrix,
                                double *emission_table,
                                double *prev_probs,
                                int current_obs,
                                double *curr_probs,
                                int *backpaths);

__device__ void max(double const *vals, int n, double *max, int *idx_max);

__host__ int *viterbi_cuda(int n_states,
                           int n_possible_observations,
                           int const *actual_observations,
                           int n_actual_observations,
                           double const *init_probabilities,
                           double **transition_matrix,
                           double **emission_table,
                           double ***probs_out) {
    // allocate buffers on host to store backpaths and most likely path
    int *backpaths = (int *) calloc(n_actual_observations * n_states,
                                    sizeof *backpaths);
    int *optimal_path = (int *) malloc(n_actual_observations *
                                       sizeof *optimal_path);
    assert(backpaths && optimal_path);

    // allocate buffers on device for prev/curr state probs and curr backpaths
    double *gpu_prev_probs = NULL, *gpu_curr_probs = NULL;
    int *gpu_backpaths = NULL;
    cudaMalloc(&gpu_prev_probs, n_states * sizeof *gpu_prev_probs);
    cudaMalloc(&gpu_curr_probs, n_states * sizeof *gpu_curr_probs);
    cudaMalloc(&gpu_backpaths,
               n_states * n_actual_observations * sizeof *gpu_backpaths);
    assert(gpu_prev_probs && gpu_curr_probs && gpu_backpaths);

    // allocate buffer on device to store observation for each iteration
    int *gpu_actual_obs = NULL;
    cudaMalloc(&gpu_actual_obs, sizeof *gpu_actual_obs);
    assert(gpu_actual_obs);
    cudaMemcpy(gpu_actual_obs,
               &actual_observations,
               n_actual_observations * sizeof gpu_actual_obs,
               cudaMemcpyHostToDevice);

    // allocate buffers on device for HMM params and copy from host
    int *gpu_n_states, *gpu_n_possible_obs;
    cudaMalloc(&gpu_n_states, sizeof *gpu_n_states);
    cudaMalloc(&gpu_n_possible_obs, sizeof *gpu_n_possible_obs);
    cudaMemcpy(gpu_n_states, &n_states, sizeof n_states,
               cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_n_possible_obs,
               &n_possible_observations,
               sizeof n_possible_observations,
               cudaMemcpyHostToDevice);

    double *gpu_trans;
    cudaMalloc(&gpu_trans, n_states * n_states * sizeof *gpu_trans);
    for (int i = 0; i < n_states; i++)
        cudaMemcpy(gpu_trans + i * n_states, transition_matrix[i],
                   n_states * sizeof *gpu_trans, cudaMemcpyHostToDevice);

    double *gpu_emission;
    cudaMalloc(&gpu_emission,
               n_states * n_possible_observations * sizeof *gpu_emission);
    for (int i = 0; i < n_states; i++)
        cudaMemcpy(gpu_emission + i * n_possible_observations,
                   emission_table[i],
                   n_possible_observations * sizeof *gpu_emission,
                   cudaMemcpyHostToDevice);

    // calculate initial state probabilities and copy to device memory
    double **prob_matrix = (double **) malloc(
            n_actual_observations * sizeof(double *));
    assert(prob_matrix);
    for (int i = 0; i < n_actual_observations; i++) {
        prob_matrix[i] = (double *) malloc(n_states * sizeof(double));
        assert(prob_matrix[i]);
    }
    for (int i = 0; i < n_states; i++)
        prob_matrix[0][i] = init_probabilities[i] +
                            emission_table[i][actual_observations[0]];

    cudaMemcpy(gpu_prev_probs, prob_matrix[0], n_states * sizeof prob_matrix[0],
               cudaMemcpyHostToDevice);


    // calculate state probabilities for subsequent observations (parallel)
    int *gpu_current_state;
    cudaMalloc(&gpu_current_state, sizeof *gpu_current_state);

    u_int64_t start = TimeStamp();
    for (int i = 1; i < n_actual_observations; i++) {
        max_probability << < n_states, THREADS_PER_BLOCK,
                n_states * 3 * sizeof(double) +
                THREADS_PER_BLOCK * (sizeof(int) + sizeof(double)) >> >
                (gpu_n_states,
                        gpu_n_possible_obs,
                        gpu_trans,
                        gpu_emission,
                        gpu_prev_probs,
                        actual_observations[i],
                        gpu_curr_probs,
                        gpu_backpaths + i * n_actual_observations);
//        memcpy(backpaths + i * n_actual_observations, paths_last,
//               n_states * sizeof *paths_last);
#ifdef DEBUG
        printf("T%d: ", i-1);
        for (int j = 0; j < n_states; ++j) {
            printf("%d ", backpaths[(i-1) * n_actual_observations + j]);
        }
        putchar('\n');
        printf("T%d: ", i);
        for (int j = 0; j < n_states; ++j) {
            printf("%d ", backpaths[i * n_actual_observations + j]);
        }
        putchar('\n');

#endif
        double *temp = (double *) malloc(n_states * sizeof *temp);
        assert(temp);
        cudaMemcpy(temp, gpu_curr_probs, n_states * sizeof *temp,
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(gpu_prev_probs, temp, n_states * sizeof *temp,
                   cudaMemcpyHostToDevice);
        memcpy(prob_matrix[i], temp, n_states * sizeof *temp);
    }
    cudaMemcpy(backpaths, gpu_backpaths,
               n_states * n_actual_observations * sizeof *backpaths,
               cudaMemcpyDeviceToHost);

#ifdef DEBUG
    printf("[CUDA PROBS TABLE]\n");
    printf("    ");
    for ( int i=0;i<n_actual_observations;i++){
        printf("T%d          ",i);
    }
    printf("\n");

    printf("OBS:");
    for ( int i=0;i<n_actual_observations;i++){
        printf(" %d          ",actual_observations[i]);
    }
    printf("\n");
    for (int i = 0; i < n_states; i++) {
        printf("S%d: ",i);
        for (int j = 0; j < n_actual_observations; j++) {
            printf("%.4e ", prob_matrix[j][i]);
        }
        putchar('\n');
    }
#endif
    // determine highest final state probability
    double *temp = (double *) malloc(n_states * sizeof *temp);
    assert(temp);
    cudaMemcpy(temp, gpu_prev_probs, n_states * sizeof *temp,
               cudaMemcpyDeviceToHost);
    double max_prob = -DBL_MAX;
    for (int i = 0; i < n_states; i++) {
        if (temp[i] > max_prob) {
            max_prob = temp[i];
            optimal_path[n_actual_observations - 1] = i;
        }
    }

    // follow backpaths to determine all states
    for (int i = n_actual_observations - 1; i > 0; i--)
        optimal_path[i - 1] = backpaths[i * n_actual_observations
                                        + optimal_path[i]];

#ifdef DEBUG
    printf("[ CUDA BACKPATHS TABLE ]\n");
    printf("    ");
    for ( int i=0;i<n_actual_observations;i++){
        printf("T%d ",i);
    }
    printf("\n");
    for (int i = 0; i < n_states; i++) {
        printf("S%d: ",i);
        for (int j = 0; j < n_actual_observations; j++) {
            printf("%2d ", backpaths[j * n_actual_observations + i]);
        }
        putchar('\n');
    }
#endif
    *probs_out = prob_matrix;

    free(temp);
//    free(backpaths);
    cudaFree(gpu_prev_probs);
    cudaFree(gpu_curr_probs);
    cudaFree(gpu_backpaths);
    cudaFree(gpu_actual_obs);
    cudaFree(gpu_n_states);
    cudaFree(gpu_n_possible_obs);
    cudaFree(gpu_trans);
    cudaFree(gpu_emission);
    cudaFree(gpu_current_state);
    return optimal_path;
}

__global__ void max_probability(int *n_states,
                                int *n_possible_observations,
                                double *transition_matrix,
                                double *emission_matrix,
                                double *gpu_prev_probs,
                                int current_obs,
                                double *gpu_curr_probs,
                                int *gpu_backpaths) {

    extern __shared__ double shared_bank[];
    int const bidx = blockIdx.x;
    int const tidx = threadIdx.x;

    int dev_n_states = *n_states;
    int dev_n_possible_obs = *n_possible_observations;
    double dev_emi_cell = emission_matrix[bidx * dev_n_possible_obs +
                                          current_obs];

    double *dev_tran_matrix = shared_bank;
    double *dev_prev_probs = dev_tran_matrix + *n_states;
    double *dev_curr_probs = dev_prev_probs + *n_states;

    double *dev_max_probs = dev_curr_probs + *n_states;
    int *dev_max_indices = (int *) (dev_max_probs + blockDim.x);

    if (tidx == 0) {
        for (int j = 0; j < dev_n_states; ++j) {
            dev_tran_matrix[j] = transition_matrix[j * dev_n_states + bidx];
            dev_prev_probs[j] = gpu_prev_probs[j];
        }
    }
    __syncthreads();

    int chunk_size = ceil(__int2double_rn(*n_states) / blockDim.x);
    int offset = tidx * chunk_size, i_max = 0;
    double p_max = -DBL_MAX, p_temp;

    for (int i = offset; i < *n_states && i < offset + chunk_size; i++) {
        p_temp = dev_prev_probs[i] + dev_tran_matrix[i] + dev_emi_cell;
        if (p_temp > p_max) {
            p_max = p_temp;
            i_max = i;
        }
    }

    dev_max_probs[tidx] = p_max;
    dev_max_indices[tidx] = i_max;
    __syncthreads();

    if (tidx == 0) {
        p_max = -DBL_MAX;
        for (int i = 0; i < blockDim.x && i < dev_n_states; i++) {
            if (dev_max_probs[i] > p_max) {
                p_max = dev_max_probs[i];
                i_max = dev_max_indices[i];
            }
        }
        gpu_curr_probs[bidx] = p_max;
        gpu_backpaths[bidx] = i_max;
    }
    __syncthreads();
}

__device__ void max(double const *vals, int n, double *max, int *idx_max) {
    *max = -DBL_MAX;
    for (int i = 0; i < n; i++)
        if (vals[i] > *max) {
            *max = vals[i];
            *idx_max = i;
        }
}