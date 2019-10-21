#include "../headers/viterbi_sequential.h"
#include <float.h>

double max(double const *probs_column, int n, int *max_idx);

int *viterbi_sequential(int n_states, int n_possible_observations,
                        const int *actual_observations,
                        int n_actual_observations,
                        const double *start_probabilities,
                        double **transition_matrix,
                        double **emission_matrix) {
    // allocate memory to store probabilities and back-paths
    double **prob_matrix = (double **) malloc(
            n_actual_observations * sizeof(double *));
    int **backpaths = (int **) malloc(n_actual_observations * sizeof(int *));
    assert(prob_matrix && backpaths);
    for (int i = 0; i < n_actual_observations; i++) {
        prob_matrix[i] = (double *) malloc(n_states * sizeof(double));
        backpaths[i] = (int *) malloc(n_states * sizeof(int));
        assert(prob_matrix[i] && backpaths[i]);
    }

    // buffer to store temporary values
    double *temp = (double *) malloc(n_states * sizeof(double));
    assert(temp);

    // allocate memory to store final path
    int *optimal_path = (int *) malloc(n_actual_observations * sizeof(int));
    assert(optimal_path);

    // calculate state probabilities for initial observation
    for (int i = 0; i < n_states; i++) {
        prob_matrix[0][i] = start_probabilities[i] +
                            emission_matrix[i][actual_observations[0]];
    }

    // calculate state probabilities for subsequent actual_observations
    for (int i = 1; i < n_actual_observations; i++) {
        // calculate max probability of current observation for each state
        for (int j = 0; j < n_states; j++) {
            // calculate the probability for all possibilities of prev. state
            for (int k = 0; k < n_states; k++) {
                temp[k] = prob_matrix[i - 1][k] + transition_matrix[k][j] +
                          emission_matrix[j][actual_observations[i]];
            }
            // store the max probability and associated prev. state
            prob_matrix[i][j] = max(temp, n_states, backpaths[i] + j);
        }
    }

    // determine most probable final state
    max(prob_matrix[n_actual_observations - 1], n_states,
        optimal_path + n_actual_observations - 1);

    // follow back-paths to get most likely sequence
    for (int i = n_actual_observations - 1; i > 0; i--)
        optimal_path[i - 1] = backpaths[i][optimal_path[i]];

#ifdef DEBUG
    printf("[ SEQUENTIAL PROBS TABLE ]\n");
    printf("    ");
    for ( int i=0;i<n_actual_observations;i++){
        printf("T%d          ",i);
    }
    printf("\n");

    printf("OBS:");
    for ( int i=0;i<n_actual_observations;i++){
        printf(" %d          ",actual_observations[i]);
    }
    printf("\n");
    for (int i = 0; i < n_states; i++) {
        printf("S%d: ",i);
        for (int j = 0; j < n_actual_observations; j++) {
            printf("%.4e ", prob_matrix[j][i]);
        }
        putchar('\n');
    }
#endif // DEBUG

#ifdef DEBUG
    printf("[ SEQUENTIAL BACKPATHS TABLE ]\n");
    printf("    ");
    for ( int i=0;i<n_actual_observations;i++){
        printf("T%d ",i);
    }
    printf("\n");

    for (int i=0;i<n_states;i++){
        printf("S%d: ",i);
        for (int j=0; j< n_actual_observations; j++ ){
            printf("%2d ",backpaths[j][i]);
        }
        printf("\n");
    }
#endif

    // free memory no longer required
    for (int i = 0; i < n_actual_observations; i++) {
        free(prob_matrix[i]);
        free(backpaths[i]);
    }
    free(prob_matrix);
    free(backpaths);
    free(temp);

    return optimal_path;
}

double max(double const *probs_column, int n, int *max_idx) {
    double max_prob = -DBL_MAX;
    for (int i = 0; i < n; i++)
        if (probs_column[i] > max_prob) {
            max_prob = probs_column[i];
            *max_idx = i;
        }
    return max_prob;
}