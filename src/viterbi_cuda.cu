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
    double *gpu_trans;
    double *gpu_emission;
    int *gpu_actual_obs;
    int *gpu_backpaths;

    cudaMalloc(&gpu_trans, n_states * n_states * sizeof *gpu_trans);
    cudaMalloc(&gpu_emission,
               n_states * n_possible_obs * sizeof *gpu_emission);
    cudaMalloc(&gpu_actual_obs, n_actual_obs * sizeof *gpu_actual_obs);
    cudaMalloc(&gpu_backpaths,
               n_states * n_actual_obs * sizeof *gpu_backpaths);

    cudaMemcpy(gpu_actual_obs,
               actual_obs,
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

    size_t shared_mem_size = (THREADS_PER_BLOCK + 2 * n_states) * sizeof double
                              + (THREADS_PER_BLOCK + n_states) * sizeof int;
    viterbi_kernel<<<n_states, THREADS_PER_BLOCK, shared_mem_size>>>
            (n_states,
            n_possible_obs,
            n_actual_obs,
            gpu_trans,
            gpu_emission,
            gpu_actual_obs,
            init_probs,
            optimal_path);

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

    // pointer offsets into shared memory
    double *dev_trans_probs = shared_bank;
    double *dev_prev_probs = dev_tran_matrix + n_states;
    double *dev_max_probs = dev_prev_probs + n_states;
    int *dev_max_indices = (int *) (dev_max_probs + blockDim.x);
    int *dev_backpaths = dev_max_indices + blockDim.x;

    int chunk_size = ceil(__int2double_rn(n_states) / blockDim.x);
    int offset = tidx * chunk_size;

    // all threads calculate init state probabilities for its block
    for (int j = offset; j < n_states && j < offset + chunk_size; j++) {
        dev_prev_probs[j] = init_probs[j] +
                            emission_matrix[j * dev_n_possible_obs +
                                            actual_obs[0]];
    }

    for (int i = 1; i < n_actual_obs; i++) {
        // all threads copy in parallel from global to shared
        for (int j = offset; j < n_states && j < offset + chunk_size; j++) {
            dev_prev_probs[j] = dev_prev_probs[j] +
                                emission_matrix[j * dev_n_possible_obs +
                                                actual_obs[0]];
            dev_trans_probs[j] = trans_matrix[j * dev_n_states + bidx];
        }
        double dev_emi_cell;
        if (tidx == 0)
            dev_emi_cell = emission_matrix[bidx * n_possible_obs
                                           + actual_obs[i]];
        __syncthreads();

        // each thread in the block determines the max prob
        double p_max = -DBL_MAX, p_temp; int i_max;
        for (int j = offset; j < dev_n_states && j < offset + chunk_size; j++) {
            p_temp = dev_prev_probs[j] + dev_tran_matrix[j] + dev_emi_cell;
            if (p_temp > p_max) {
                p_max = p_temp;
                i_max = j;
            }
        }
        dev_max_probs[tidx] = p_max;
        dev_max_indices[tidx] = i_max;
        __syncthreads();

        // thread 0 calculates max prob for cell being calculated by this block
        if (tidx == 0) {
            p_max = -DBL_MAX;
            for (int j = 0; j < blockDim.x && j < dev_n_states; j++) {
                if (dev_max_probs[j] > p_max) {
                    p_max = dev_max_probs[j];
                    i_max = dev_max_indices[j];
                }
            }
            dev_prev_probs[bidx] = p_max;
            dev_backpaths[bidx] = i_max;
            // need to copy dev_backpaths to correct column in global
        }
        g.sync();
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