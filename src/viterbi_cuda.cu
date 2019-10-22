#include <stdio.h>
#include <assert.h>
#include <float.h>
#include <cooperative_groups.h>

using namespace
cooperative_groups;
#define THREADS_PER_BLOCK 32

__global__ void viterbi_kernel(int n_states,
                               int n_possible_obs,
                               int n_actual_obs,
                               double *trans_matrix,
                               double *emission_matrix,
                               int *actual_obs,
                               double *init_probs,
                               int *optimal_path
);


__device__ void max(double const *vals, int n, double *max, int *idx_max);

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
    assert(backpaths && optimal_path);

    // allocate gpu memory
    int *gpu_n_states, *gpu_n_possible_obs, *gpu_n_actual_obs;
    double *gpu_trans;
    double *gpu_emission;
    int *gpu_actual_obs;
    int *gpu_backpaths;


    cudaMalloc(&gpu_n_states, sizeof *gpu_n_states);
    cudaMalloc(&gpu_n_possible_obs, sizeof *gpu_n_possible_obs);
    cudaMalloc(&gpu_n_actual_obs, sizeof *gpu_n_actual_obs);

    cudaMalloc(&gpu_trans, n_states * n_states * sizeof *gpu_trans);
    cudaMalloc(&gpu_emission,
               n_states * n_possible_obs * sizeof *gpu_emission);
    cudaMalloc(&gpu_actual_obs, n_actual_obs * sizeof *gpu_actual_obs);
    cudaMalloc(&gpu_backpaths,
               n_states * n_actual_obs * sizeof *gpu_backpaths);

    // allocate buffers on device for HMM params and copy from host
    cudaMemcpy(gpu_n_states,
               &n_states,
               sizeof n_states,
               cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_n_possible_obs,
               &n_possible_obs,
               sizeof n_possible_obs,
               cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_n_actual_obs,
               &n_actual_obs,
               sizeof n_actual_obs,
               cudaMemcpyHostToDevice);

    cudaMemcpy(gpu_actual_obs,
               &actual_obs,
               n_actual_obs * sizeof gpu_actual_obs,
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

    viterbi_kernel << < n_states, THREADS_PER_BLOCK,
            n_states * 3 * sizeof(double) +
            THREADS_PER_BLOCK * (sizeof(int) + sizeof(double)
                                 + n_states * n_actual_obs * sizeof(int)) >> >
            (n_states,
                    n_possible_obs,
                    n_actual_obs,
                    gpu_trans,
                    gpu_emission,
                    gpu_actual_obs,
                    init_probs,
                    optimal_path);

    cudaMemcpy(backpaths, gpu_backpaths,
               n_states * n_actual_obs * sizeof *backpaths,
               cudaMemcpyDeviceToHost);

#ifdef DEBUG
    //    printf("[CUDA PROBS TABLE]\n");
    //    printf("    ");
    //    for ( int i=0;i<n_actual_obs;i++){
    //        printf("T%d          ",i);
    //    }
    //    printf("\n");
    //
    //    printf("OBS:");
    //    for ( int i=0;i<n_actual_obs;i++){
    //        printf(" %d          ",actual_obs[i]);
    //    }
    //    printf("\n");
    //    for (int i = 0; i < n_states; i++) {
    //        printf("S%d: ",i);
    //        for (int j = 0; j < n_actual_obs; j++) {
    //            printf("%.4e ", prob_matrix[j][i]);
    //        }
    //        putchar('\n');
    //    }
#endif
    // determine highest final state probability
//    double *temp = (double *) malloc(n_states * sizeof *temp);
//    assert(temp);
//    cudaMemcpy(temp, gpu_prev_probs, n_states * sizeof *temp,
//               cudaMemcpyDeviceToHost);
//    double max_prob = -DBL_MAX;
//    for (int i = 0; i < n_states; i++) {
//        if (temp[i] > max_prob) {
//            max_prob = temp[i];
//            optimal_path[n_actual_obs - 1] = i;
//        }
//    }
//
//    // follow backpaths to determine all states
//    for (int i = n_actual_obs - 1; i > 0; i--)
//        optimal_path[i - 1] = backpaths[i * n_actual_obs
//                                        + optimal_path[i]];

//#ifdef DEBUG
//    printf("[ CUDA BACKPATHS TABLE ]\n");
//    printf("    ");
//    for ( int i=0;i<n_actual_obs;i++){
//        printf("T%d ",i);
//    }
//    printf("\n");
//    for (int i = 0; i < n_states; i++) {
//        printf("S%d: ",i);
//        for (int j = 0; j < n_actual_obs; j++) {
//            printf("%2d ", backpaths[j * n_actual_obs + i]);
//        }
//        putchar('\n');
//    }
//#endif
//
//    free(temp);
//    free(backpaths);
//
//    cudaFree(gpu_curr_probs);
//    cudaFree(gpu_backpaths);
//    cudaFree(gpu_actual_obs);
//    cudaFree(gpu_n_states);
//    cudaFree(gpu_n_possible_obs);
//    cudaFree(gpu_trans);
//    cudaFree(gpu_emission);
//    cudaFree(gpu_current_state);
    return optimal_path;
}

__global__ void viterbi_kernel(int n_states,
                               int n_possible_obs,
                               int n_actual_obs,
                               double *trans_matrix,
                               double *emission_matrix,
                               int *actual_obs,
                               double *init_probs,
                               int *optimal_path) {

    multi_grid_group g = this_multi_grid();
    extern __shared__ double shared_bank[];
    int const bidx = blockIdx.x;
    int const tidx = threadIdx.x;

    int dev_n_states = n_states;
    int dev_n_possible_obs = n_possible_obs;

    double *dev_tran_matrix = shared_bank;
    double *dev_prev_probs = dev_tran_matrix + n_states;
    double *dev_curr_probs = dev_prev_probs + n_states;

    double *dev_max_probs = dev_curr_probs + n_states;
    int *dev_max_indices = (int *) (dev_max_probs + blockDim.x);

    int *dev_backpaths = dev_max_indices + blockDim.x;

    int chunk_size = ceil(__int2double_rn(n_states) / blockDim.x);
    int offset = tidx * chunk_size, i_max = 0;
    double p_max = -DBL_MAX, p_temp;

    for (int i = offset; i < n_states && i < offset + chunk_size; i++) {
        dev_prev_probs[i] = init_probs[i] +
                            emission_matrix[i * dev_n_possible_obs +
                                            actual_obs[0]];
        dev_tran_matrix[i] = trans_matrix[i * dev_n_states + bidx];
    }
    g.sync();

    for (int j = 0; j < n_actual_obs; ++j) {
        double dev_emi_cell = 4.2;
        for (int i = offset; i < dev_n_states && i < offset + chunk_size; i++) {
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
            dev_prev_probs[bidx] = p_max;
            dev_backpaths[j * dev_n_states + bidx] = i_max;
        }
        g.sync();
    }

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