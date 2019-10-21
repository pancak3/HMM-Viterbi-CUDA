#include <stdio.h>
#include <assert.h>
#include <float.h>

__global__ void max_probability(int *n_states,
                                int *n_possible_observations,
                                double *transition_matrix,
                                double *emission_table,
                                double *prev_probs,
                                int *current_state,
                                double *curr_probs,
                                int *backpaths);

__device__ void max(double const *vals, int n, double *max, int *idx_max);

__host__ int *viterbi_cuda(int n_states,
                           int n_possible_observations,
                           int const *actual_observations,
                           int n_actual_observations,
                           double const *init_probabilities,
                           double **transition_matrix,
                           double **emission_table) {
    // allocate buffers on host to store backpaths and most likely path
    int *backpaths = (int *) calloc(n_actual_observations * n_states,
                                    sizeof *backpaths);
    int *optimal_path = (int *) malloc(n_actual_observations *
                                       sizeof *optimal_path);
    assert(backpaths && optimal_path);

    // allocate buffers on device for prev/curr state probs and curr backpaths
    double *dev_prev_probs = NULL, *dev_curr_probs = NULL;
    int *dev_backpaths = NULL;
    cudaMalloc(&dev_prev_probs, n_states * sizeof *dev_prev_probs);
    cudaMalloc(&dev_curr_probs, n_states * sizeof *dev_curr_probs);
    cudaMalloc(&dev_backpaths, n_states * sizeof *dev_backpaths);
    assert(dev_prev_probs && dev_curr_probs && dev_backpaths);

    // allocate buffer on device to store observation for each iteration
    int *dev_actual_obs = NULL;
    cudaMalloc(&dev_actual_obs, sizeof *dev_actual_obs);
    assert(dev_actual_obs);
    cudaMemcpy(dev_actual_obs,
               &actual_observations,
               n_actual_observations * sizeof dev_actual_obs,
               cudaMemcpyHostToDevice);

    // allocate buffers on device for HMM params and copy from host
    int *dev_n_states, *dev_n_possible_obs;
    cudaMalloc(&dev_n_states, sizeof *dev_n_states);
    cudaMalloc(&dev_n_possible_obs, sizeof *dev_n_possible_obs);
    cudaMemcpy(dev_n_states, &n_states, sizeof n_states,
               cudaMemcpyHostToDevice);
    cudaMemcpy(dev_n_possible_obs,
               &n_possible_observations,
               sizeof n_possible_observations,
               cudaMemcpyHostToDevice);

    double *dev_trans;
    cudaMalloc(&dev_trans, n_states * n_states * sizeof *dev_trans);
    for (int i = 0; i < n_states; i++)
        cudaMemcpy(dev_trans + i * n_states, transition_matrix[i],
                   n_states * sizeof *dev_trans, cudaMemcpyHostToDevice);

    double *dev_emission;
    cudaMalloc(&dev_emission,
               n_states * n_possible_observations * sizeof *dev_emission);
    for (int i = 0; i < n_states; i++)
        cudaMemcpy(dev_emission + i * n_possible_observations,
                   emission_table[i],
                   n_possible_observations * sizeof *dev_emission,
                   cudaMemcpyHostToDevice);

    // calculate initial state probabilities and copy to device memory
    double *temp = (double *) malloc(n_states * sizeof *temp);
    assert(temp);
    for (int i = 0; i < n_states; i++)
        temp[i] = init_probabilities[i] +
                  emission_table[i][actual_observations[0]];
    cudaMemcpy(dev_prev_probs, temp, n_states * sizeof *temp,
               cudaMemcpyHostToDevice);
#ifdef DEBUG
    double **prob_matrix = (double **) malloc(
            n_actual_observations * sizeof(double *));
    assert(prob_matrix);
    for (int i = 0; i < n_actual_observations; i++) {
        prob_matrix[i] = (double *) malloc(n_states * sizeof(double));
        assert(prob_matrix[i] );
    }
    for (int i = 0; i < n_states; i++) {
        prob_matrix[0][i] = temp[i];
    }
#endif

    // calculate state probabilities for subsequent observations (parallel)
    int *dev_current_state;
    cudaMalloc(&dev_current_state, sizeof *dev_current_state);
    for (int i = 1; i < n_actual_observations; i++) {
        cudaMemcpy(dev_current_state, actual_observations + i,
                   sizeof *dev_current_state, cudaMemcpyHostToDevice);
        max_probability << < n_states, 32 >> > (dev_n_states,
                dev_n_possible_obs,
                dev_trans,
                dev_emission,
                dev_prev_probs,
                dev_current_state,
                dev_curr_probs,
                dev_backpaths);
        cudaMemcpy(backpaths + i * n_actual_observations,
                   dev_backpaths,
                   n_states * sizeof *backpaths,
                   cudaMemcpyDeviceToHost);
        for (int j = 0; j < n_states; ++j) {
            printf("%d ", backpaths[i * n_actual_observations + j]);
        }
        putchar('\n');
        // swap probs;
#ifndef DEBUG
        temp = dev_prev_probs;
        dev_prev_probs = dev_curr_probs;
        dev_curr_probs = temp;
#endif
#ifdef DEBUG
        cudaMemcpy(temp, dev_curr_probs, n_states * sizeof *temp,
                cudaMemcpyDeviceToHost);
        cudaMemcpy(dev_prev_probs, temp, n_states * sizeof *temp,
                cudaMemcpyHostToDevice);
        memcpy(prob_matrix[i], temp, n_states * sizeof *temp);
#endif
    }
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
#ifndef DEBUG
    cudaMemcpy(temp, dev_prev_probs, n_states * sizeof *temp,
               cudaMemcpyDeviceToHost);
#endif
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

    free(temp);
    free(backpaths);
    cudaFree(dev_prev_probs);
    cudaFree(dev_curr_probs);
    cudaFree(dev_backpaths);
    cudaFree(dev_actual_obs);
    cudaFree(dev_n_states);
    cudaFree(dev_n_possible_obs);
    cudaFree(dev_trans);
    cudaFree(dev_emission);
    cudaFree(dev_current_state);
    return optimal_path;
}

__global__ void max_probability(int *n_states,
                                int *n_possible_observations,
                                double *transition_matrix,
                                double *emission_matrix,
                                double *prev_probs,
                                int *current_state,
                                double *curr_probs,
                                int *backpaths) {
    int const bidx = blockIdx.x;
    int const tidx = threadIdx.x;

    if (tidx == 0) {
        double p_max = -DBL_MAX, p_temp;
        int i_max = -1;

        for (int i = 0; i < *n_states; i++) {
            p_temp = prev_probs[i] +
                     transition_matrix[i * *n_states + bidx] +
                     emission_matrix[bidx * *n_possible_observations +
                                     *current_state];
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