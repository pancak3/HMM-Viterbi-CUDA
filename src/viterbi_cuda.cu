#include <stdio.h>
#include <assert.h>
#include <float.h>

#define THREADS_PER_BLOCK 32

__global__ void back_track_kernel(double *probs, int n_threads, int *out_idx);

__global__ void viterbi_kernel(int n_states,
                               int n_possible_obs,
                               int n_actual_obs,
                               double *trans_matrix,
                               double *emission_matrix,
                               int *global_actual_obs,
                               double *global_probs,
                               int *global_backpaths,
                               int current_t);

__host__ int *viterbi_cuda(int n_states,
                           int n_possible_obs,
                           int n_actual_obs,
                           int *actual_obs,
                           double *init_probs,
                           double **transition_matrix,
                           double **emission_matrix) {

    int *backpaths = (int *) calloc(n_actual_obs * n_states,
                                    sizeof *backpaths);
    int *optimal_path = (int *) malloc(n_actual_obs *
                                       sizeof *optimal_path);
    double *probs = (double *) calloc(n_states * n_actual_obs, sizeof(double));

    assert(backpaths && optimal_path && probs);

    // allocate gpu memory
    double *gpu_trans;
    double *gpu_emission;
    double *gpu_probs;

    int *gpu_actual_obs;
    int *gpu_backpaths;

    cudaMalloc(&gpu_trans, n_states * n_states * sizeof *gpu_trans);
    cudaMalloc(&gpu_emission,
               n_states * n_possible_obs * sizeof *gpu_emission);
    cudaMalloc(&gpu_probs,
               n_states * n_actual_obs * sizeof *gpu_probs);

    cudaMalloc(&gpu_actual_obs, n_actual_obs * sizeof *gpu_actual_obs);
    cudaMalloc(&gpu_backpaths,
               n_states * n_actual_obs * sizeof *gpu_backpaths);

    cudaMemcpy(gpu_actual_obs,
               actual_obs,
               n_actual_obs * sizeof *gpu_actual_obs,
               cudaMemcpyHostToDevice);

    for (int i = 0; i < n_states; i++) {
        cudaMemcpy(gpu_trans + i * n_states,
                   transition_matrix[i],
                   n_states * sizeof *gpu_trans,
                   cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_emission + i * n_possible_obs,
                   emission_matrix[i],
                   n_possible_obs * sizeof *gpu_emission,
                   cudaMemcpyHostToDevice);
    }

    // calc first T probs then copy to gpu
    for (int i = 0; i < n_states; i++) {
        probs[i] = init_probs[i] +
                   emission_matrix[i][actual_obs[0]];
    }
    cudaMemcpy(gpu_probs,
               probs,
               n_states * sizeof *gpu_probs,
               cudaMemcpyHostToDevice);

    size_t shared_mem_size = (THREADS_PER_BLOCK + n_states) * sizeof(int);
    shared_mem_size += (THREADS_PER_BLOCK + 2 * n_states) * sizeof(double);

    for (int i = 1; i < n_actual_obs; ++i) {
        viterbi_kernel << < n_states, THREADS_PER_BLOCK, shared_mem_size >> >
                                                         (n_states,
                                                                 n_possible_obs,
                                                                 n_actual_obs,
                                                                 gpu_trans,
                                                                 gpu_emission,
                                                                 gpu_actual_obs,
                                                                 gpu_probs,
                                                                 gpu_backpaths,
                                                                 i);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(probs, gpu_probs, n_states * n_actual_obs * sizeof(double),
               cudaMemcpyDeviceToHost);

    cudaMemcpy(backpaths, gpu_backpaths, n_states * n_actual_obs * sizeof(int),
               cudaMemcpyDeviceToHost);

    // determine most probable final state
    double max_prob = -DBL_MAX;
    int max_idx = -1;
    for (int i = 0; i < n_states; i++)
        if (probs[(n_actual_obs - 1) * n_states + i] > max_prob) {
            max_prob = probs[(n_actual_obs - 1) * n_states + i];
            max_idx = i;
        }
    optimal_path[n_actual_obs - 1] = max_idx;

    // follow back-paths to get most likely sequence
    for (int i = n_actual_obs - 1; i > 0; i--)
        optimal_path[i - 1] = backpaths[i * n_states + optimal_path[i]];

#ifdef DEBUG
    printf("[ CUDA PROBS TABLE ]\n");
    printf("    ");
    for (int i = 0; i < n_actual_obs; i++) {
        printf("T%d          ", i);
    }
    printf("\n");

    printf("OBS:");
    for (int i = 0; i < n_actual_obs; i++) {
        printf(" %d          ", actual_obs[i]);
    }
    printf("\n");
    for (int i = 0; i < n_states; i++) {
        printf("S%d: ", i);
        for (int j = 0; j < n_actual_obs; j++) {
            printf("%.4e ", probs[j * n_states + i]);
        }
        putchar('\n');
    }
#endif // DEBUG

#ifdef DEBUG
    printf("[ CUDA BACKPATHS TABLE ]\n");
    printf("    ");
    for ( int i=0;i<n_actual_obs;i++){
        printf("T%d ",i);
    }
    printf("\n");
    for (int i = 0; i < n_states; i++) {
        printf("S%d: ",i);
        for (int j = 0; j < n_actual_obs; j++) {
            printf("%2d ", backpaths[j *n_states  + i]);
        }
        putchar('\n');
    }
#endif

    free(backpaths);
    free(probs);

    cudaFree(gpu_trans);
    cudaFree(gpu_emission);
    cudaFree(gpu_probs);
    cudaFree(gpu_actual_obs);
    cudaFree(gpu_backpaths);
    return optimal_path;
}

__global__ void viterbi_kernel(int n_states,
                               int n_possible_obs,
                               int n_actual_obs,
                               double *trans_matrix,
                               double *emission_matrix,
                               int *global_actual_obs,
                               double *global_probs,
                               int *global_backpaths,
                               int current_t) {

    extern __shared__ double shared_bank[];
    int const bidx = blockIdx.x;
    int const tidx = threadIdx.x;

    // pointer offsets into shared memory
    double *dev_trans_probs = shared_bank;
    double *dev_prev_probs = dev_trans_probs + n_states;
    double *dev_max_probs = dev_prev_probs + n_states;
    int *dev_max_indices = (int *) (dev_max_probs + blockDim.x);

    int chunk_size = ceil(__int2double_rn(n_states) / blockDim.x);
    int offset = tidx * chunk_size;

    // all blocks calculate init state probabilities

    double dev_emission_cell = emission_matrix[bidx * n_possible_obs
                                               + global_actual_obs[current_t]];
    // all threads copy in parallel from global to shared
    for (int j = offset; j < n_states && j < offset + chunk_size; j++) {
        dev_trans_probs[j] = trans_matrix[j * n_states + bidx];
    }
    __syncthreads();

    // each thread in the block determines the max prob
    double p_max = -DBL_MAX, p_temp;
    int i_max;
    for (int j = offset; j < n_states && j < offset + chunk_size; j++) {
        p_temp = global_probs[(current_t - 1) * n_states + j] +
                 dev_trans_probs[j] +
                 dev_emission_cell;
        if (p_temp > p_max) {
            p_max = p_temp;
            i_max = j;
        }
    }
    dev_max_probs[tidx] = p_max;
    dev_max_indices[tidx] = i_max;
    __syncthreads();

    // thread 0 find the max prob for cell of its block
    if (tidx == 0) {
        p_max = -DBL_MAX;
        for (int j = 0; j < blockDim.x && j < n_states; j++) {
            if (dev_max_probs[j] > p_max) {
                p_max = dev_max_probs[j];
                i_max = dev_max_indices[j];
            }
        }
        global_probs[current_t * n_states + bidx] = p_max;
        global_backpaths[current_t * n_states + bidx] = i_max;
    }
    __syncthreads();
}

__global__ void back_track_kernel(double *probs, int n_threads, int *out_idx) {

    extern __shared__ double shared_back[];

    int const tidx = threadIdx.x;

    double *max_probs = shared_back;
    int *max_idx = (int *) (max_probs + n_threads);


    int chunk_size = ceil(__int2double_rn(n_threads) / blockDim.x);
    int offset = tidx * chunk_size;
    double p_max = -DBL_MAX, p_temp;

    int i_max;
    for (int j = offset; j < offset + chunk_size; j++) {
        p_temp = probs[j];
        if (p_temp > p_max) {
            p_max = p_temp;
            i_max = j;
        }
    }
    max_probs[tidx] = p_max;
    max_idx[tidx] = i_max;
    __syncthreads();

    // thread 0 find the max prob for cell of its block
    if (tidx == 0) {
        p_max = -DBL_MAX;
        for (int j = 0; j < n_threads; j++) {
            if (max_probs[j] > p_max) {
                p_max = max_probs[j];
                i_max = max_idx[j];
            }
        }
        *out_idx = i_max;
    }
    __syncthreads();
};
