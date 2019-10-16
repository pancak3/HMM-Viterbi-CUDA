//
// Created by q on 16/10/19.
//

#include <stdio.h>
#include <stdlib.h>
#include "../headers/driver.h"

int *viterbi_sequential(int n_states, int n_observations,
                        int const *observations,
                        int observations_length,
                        double const *init_probabilities,
                        double const **transition_matrix,
                        double const **emission_table);