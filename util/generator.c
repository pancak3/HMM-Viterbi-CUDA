#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>

#define MIN_PROB 0.005
#define MAX_PROB 0.7310585

double rand_prob(double *high);

int my_rand(int min, int max);

int pancake_rand();


double prob_sum = 1.0;

double rand_prob(double *high) {
    int max = *high * 10000;
    int min = 0;
    if (max == 0) {
        return 0;
    }
    double t;

    t = (double) (my_rand(min, max)) / 10000;
    t *= (double) (my_rand(min, max)) / 10000;

    *high -= t;

    return t;
}

int my_rand(int min, int max) {
    return pancake_rand() % (max - min + 1) + min;
}

int G_RAND_SEED = 1023;

int pancake_rand() {
    int rand_num, func_seed;

    func_seed = (int) clock() + G_RAND_SEED;
    func_seed *= func_seed;
    rand_num = (int) ((524288 * func_seed + 137438953471) % 2147483647);
    G_RAND_SEED = rand_num;

    return rand_num;
}

int main() {
    int i, j;
    int states, observations, observation_length;
    // observations < (10000 / MIN_PROB = 2000)
    scanf("%d %d %d", &states, &observations, &observation_length);
    printf("%d %d\n", states, observations);

    double current_prob;
    srand(clock());

    // transition_prob
    double **transition_matrix = malloc(states * sizeof *transition_matrix);
    assert(transition_matrix);
    for (int i = 0; i < states; i++) {
        transition_matrix[i] = malloc(states * sizeof *transition_matrix[i]);
        assert(transition_matrix[i]);
    }
    for (i = 0; i < states; i++) {
        current_prob = prob_sum - states * MIN_PROB;
        for (j = 0; j < states - 1; j++) {
            transition_matrix[i][j] = rand_prob(&current_prob) + MIN_PROB;
            printf("%.4f ", transition_matrix[i][j]);
        }
        transition_matrix[i][j] = current_prob + MIN_PROB;
        printf("%.4f\n", transition_matrix[i][j]);
    }

    // calc stat prob with Bayes rule, but here raise : loss of precision
    // ->  (0.9999 <= sum(P[i]) <= 1)
    double *P = malloc(states * sizeof *P);
    double denominator = 1;
    for (int i = 1; i < states; i++) {
        denominator += transition_matrix[0][i] / transition_matrix[i][0];
    }
    P[0] = 1 / denominator;
    printf("%.4f ", P[0]);
    for (int i = 1; i < states; i++) {
        P[i] = P[0] * transition_matrix[0][i] / transition_matrix[i][0];
        printf("%.4f ", P[i]);

    }
    printf("\n");

    // emission_prob
    for (i = 0; i < states; i++) {
        current_prob = 1 - observations * MIN_PROB;
        for (j = 0; j < observations - 1; j++) {
            printf("%.4f ", rand_prob(&current_prob) + MIN_PROB);
        }
        printf("%.4f\n", current_prob + MIN_PROB);
    }

    // random obs
    printf("%d\n", observation_length);
    for (i = 0; i < observation_length; i++) {
        printf("%d\n", my_rand(0, observations - 1));
    }
    return 0;
}