//
// Created by q on 16/10/19.
//
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

double *read_init_probabilities(FILE *f, int states);

double **read_transition_matrix(FILE *f, int states);

double **read_emission_table(FILE *f, int states, int emissions);

void free_2D_memory(double **table, int rows);

int *read_observation(FILE *f, int observations);