#include <stdio.h>
#include <assert.h>
#include <float.h>

__global__ void max_probability(int n_states, int n_observations,
                                double **transition_matrix,
                                double **emission_table,
                                double *prev_probs, double *curr_probs,
                                int *backpaths);

__device__ void max(double const *vals, int n, double *max, int *idx_max);

__host__ int *viterbi_cuda(int n_states, int n_observations,
                           int const *observations, int observations_length,
                           double const *init_probabilities,
                           double **transition_matrix,
                           double **emission_table) {
    // allocate memory to store back-paths
    int *backpaths = NULL;
    cudaMalloc(&backpaths, observations_length * n_states * sizeof *backpaths);
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

    // calculate state probabilities for subsequent observations (parallel)
    for (int i = 1; i < observations_length; i++) {
        max_probability<<<56, 64>>>(n_states, n_observations,
                                    transition_matrix, emission_table,
                                    prev_probs, curr_probs,
                                    backpaths + i * n_states);
        double *temp = curr_probs;
        prev_probs = curr_probs;
        curr_probs = temp;
    }
    return NULL;
}

__global__ void max_probability(int n_states, int n_observations,
                                double **transition_matrix,
                                double **emission_table,
                                double *prev_probs, double *curr_probs,
                                int *backpaths) {
    int const bidx = blockIdx.x;
    int const tidx = threadIdx.x;

    __shared__ int n_s;
    __shared__ int n_obs;
    extern __shared__ double pre[];
    extern __shared__ double t_matrix[];
    extern __shared__ double e_table[];

    if (tidx == 0) {
        n_s = n_states;
        n_obs = n_observations;
        for (int i = 0; i < n_s; i++) {
            pre[i] = prev_probs[i];
            for (int j = 0; j < n_s; j++)
                t_matrix[i * n_s + j] = transition_matrix[i][j];
            for (int j = 0; j < n_obs; j++)
                e_table[i * n_obs + j] = emission_table[i][j];
        }
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