#include "../headers/viterbi_sequential.h"

double max(double const *prev_probs, double const **transition_matrix,
           double const **emission_table, int n_states, int n_observations,
           int curr_state, int observation, int *optimal_state);

//int main_() {
//    int states, emissions, observations;
//    scanf("%d %d %d", &states, &emissions, &observations);
//    double *init_probabilities = read_init_probabilities(stdin, states);
//    double **transition_matrix = read_transition_matrix(stdin, states);
//    double **emission_table = read_emission_table(stdin, states, emissions);
//    int *observation_table = read_observation(stdin, observations);
//    return 0;
//}

int *viterbi_sequential(int const n_states, int const n_observations,
                        int const *observations,
                        int const observations_length,
                        double const *init_probabilities,
                        double const **transition_matrix,
                        double const **emission_table) {
    int *optimal_path = malloc(n_observations * sizeof *optimal_path);
    double *prev_probs = malloc(n_states * sizeof *prev_probs);
    double *curr_probs = malloc(n_states * sizeof *curr_probs);
    assert(optimal_path && prev_probs && curr_probs);

    // calculate state probabilities for initial observation
    for (int i = 0; i < n_states; i++)
        curr_probs[i] = init_probabilities[i] *
                        emission_table[observations[0]][i];
//#ifdef DEBUG
    printf("[*] ============= \n");
    printf("[Time %2d, OBS %2d] ", 0, observations[0]);
    for (int j = 0; j < n_states; j++)
        printf("%.8lf ", curr_probs[j]);
    putchar('\n');
//#endif // DEBUG

    for (int i = 1; i < observations_length; i++) {
        // swap pointers for prev and curr probabilities
        double *temp = prev_probs;
        prev_probs = curr_probs;
        curr_probs = temp;

        for (int curr_state = 0; curr_state < n_states; curr_state++)
            curr_probs[curr_state] = max(prev_probs, transition_matrix, emission_table,
                                         n_states, n_observations, curr_state, observations[i],
                                optimal_path + i - 1);

//#ifdef DEBUG
        printf("[Time %2d, OBS %2d] ", i, observations[i]);
        for (int j = 0; j < n_states; j++)
            printf("%.8lf ", curr_probs[j]);
        putchar('\n');
//#endif // DEBUG
    }

    // calculate best option for last observation
    double max_prob = 0;
    for (int i = 0; i < n_states; i++)
        if (curr_probs[i] > max_prob) {
            max_prob = curr_probs[i];
            optimal_path[observations_length - 1] = i;
        }

    return optimal_path;
}

double max(double const *prev_probs, double const **transition_matrix,
           double const **emission_table, int n_states, int n_observations,
           int curr_state, int observation, int *optimal_state) {
    double prob, max = 0;
    for (int i = 0; i < n_states; i++) {
        // emission_table[i][observation] is not the same as input
        prob = prev_probs[i] * transition_matrix[i][curr_state] *
               emission_table[i][observation];
        if (prob > max) {
            max = prob;
            *optimal_state = i;
        }
    }
    return max;
}