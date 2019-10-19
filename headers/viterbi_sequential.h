//
// Created by q on 16/10/19.
//

#include <stdio.h>
#include <stdlib.h>
#include "../headers/driver.h"

int *viterbi_sequential(int n_states, int n_observations,
                        const int *observations,
                        int observations_length,
                        const double *init_probabilities,
                        double **transition_matrix,
                        double **emission_table);