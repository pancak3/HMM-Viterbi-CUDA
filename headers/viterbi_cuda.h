//
// Created by q on 16/10/19.
//

#include <stdio.h>
#include <stdlib.h>

__host__ int* viterbi_cuda(int n_states, int n_possible_obs, int n_actual_obs,
						   int* actual_obs, double* init_probabilities,
						   double** transition_matrix,
						   double** emission_table);