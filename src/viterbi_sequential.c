#include "../headers/viterbi_sequential.h"

int main_() {
    int states, emissions, observations;
    scanf("%d %d %d", &states, &emissions, &observations);
    double *init_probabilities = read_init_probabilities(stdin, states);
    double **transition_matrix = read_transition_matrix(stdin, states);
    double **emission_table = read_emission_table(stdin, states, emissions);
    int *observation_table = read_observation(stdin, observations);
    return 0;
}

int *viterbi(int state_num, int observation_num, double *init_probabilities, double **transition_matrix,
             double **emission_table) {
    int *optimal_path = malloc(observation_num * sizeof(optimal_path));

    int *column_prob = malloc(state_num * sizeof(column_prob));
    int i;
    for (i = 0; i < state_num; i++) {
        column_prob[i] = i;
    }
    return optimal_path;
}