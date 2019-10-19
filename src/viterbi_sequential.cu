#include <math.h>
#include "../headers/viterbi_sequential.h"

double max(double const *vals, int n, int *idx_max);

int *viterbi_sequential(int n_states, int n_observations,
                        const int *observations,
                        int observations_length,
                        const double *init_probabilities,
                        double **transition_matrix,
                        double **emission_table) {
    // allocate memory to store probabilities and back-paths
    double **probs = (double **) malloc(observations_length * sizeof(double *));
    int **backpaths = (int **) malloc(observations_length * sizeof(int *));
    assert(probs && backpaths);
    for (int i = 0; i < observations_length; i++) {
        probs[i] = (double *) malloc(n_states * sizeof(double));
        backpaths[i] = (int *) malloc(n_states * sizeof(int));
        assert(probs[i] && backpaths[i]);
    }

    // buffer to store temporary values
    double *temp = (double *) malloc(n_states * sizeof(double));
    assert(temp);

    // allocate memory to store final path
    int *optimal_path = (int *) malloc(observations_length * sizeof(int));
    assert(optimal_path);

    // calculate state probabilities for initial observation
    for (int i = 0; i < n_states; i++) {
        probs[0][i] = init_probabilities[i] +
                      emission_table[i][observations[0]];
    }

    // calculate state probabilities for subsequent observations
    for (int i = 1; i < observations_length; i++) {
        // calculate max probability of current observation for each state
        for (int j = 0; j < n_states; j++) {
            // calculate the probability for all possibilities of prev. state
            for (int k = 0; k < n_states; k++) {
                temp[k] = probs[i - 1][k] + transition_matrix[k][j] +
                          emission_table[j][observations[i]];
            }
            // store the max probability and associated prev. state
            probs[i][j] = max(temp, n_states, backpaths[i] + j);
        }
    }

    // determine most probable final state
    max(probs[observations_length - 1], n_states,
        optimal_path + observations_length - 1);

    // follow back-paths to get most likely sequence
    for (int i = observations_length - 1; i > 0; i--)
        optimal_path[i - 1] = backpaths[i][optimal_path[i]];

#ifdef DEBUG
    for (int i = 0; i < observations_length; i++) {
        printf("Time:%d OBS:%d |\t", i, observations[i]);
        for (int j = 0; j < n_states; j++) {
            printf("%.4e ", probs[i][j]);
        }
        putchar('\n');
    }
#endif // DEBUG

    // free memory no longer required
    for (int i = 0; i < observations_length; i++) {
        free(probs[i]);
        free(backpaths[i]);
    }
    free(probs);
    free(backpaths);
    free(temp);

    return optimal_path;
}

double max(double const *vals, int n, int *idx_max) {
    double max = log(0);
    for (int i = 0; i < n; i++)
        if (vals[i] > max) {
            max = vals[i];
            *idx_max = i;
        }
    return max;
}