//
// Created by q on 16/10/19.
//

#include <stdio.h>
#include <stdlib.h>
#include "../headers/driver.h"

int *viterbi(int state_num, int observation_num, double *init_probabilities, double **transition_matrix,
             double **emission_table);