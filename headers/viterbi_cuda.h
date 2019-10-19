//
// Created by q on 16/10/19.
//

#include <stdio.h>
#include <stdlib.h>

__host__ int *viterbi_cuda(int n_states, int n_observations,
                           int const *observations, int observations_length,
                           double const *init_probabilities,
                           double **transition_matrix, double **emission_table);