//
// Created by q on 16/10/19.
//

#include <stdio.h>
#include <stdlib.h>
#include "../headers/driver.h"

int *viterbi_sequential(int n_states, int n_possible_observations,
                        const int *actual_observations,
                        int n_actual_observations,
                        const double *start_probabilities,
                        double **transition_matrix,
                        double **emission_matrix);