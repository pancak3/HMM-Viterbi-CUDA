#include <stdio.h>
#include <stdlib.h>
#include "../headers/driver.h"

int main() {
    int state_num, observation_num;
    scanf("%d %d", &state_num, &observation_num);
    double *init_probabilities = read_init_probabilities(stdin, state_num);
    double **transition_matrix = read_transition_matrix(stdin, state_num);
    double **emission_table = read_emission_table(stdin, state_num, observation_num);
    return 0;
}