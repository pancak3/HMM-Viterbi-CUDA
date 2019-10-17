#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

float rand_prob(float *high);

int my_rand(int min, int max);

float rand_prob(float *high) {
    int max = *high * 10000;
    int min = 0;
    if (max == 0) {
        return 0;
    }
    float t = (float) (my_rand(min, max)) / 10000;
    *high -= t;

    return t;
}

int my_rand(int min, int max) {
    return rand() % (max - min + 1) + min;
}


int main() {
    int i, j;
    int states, observations, observation_length;
    scanf("%d %d %d", &states, &observations, &observation_length);
    printf("%d %d\n", states, observations);
    float current_prob = 1;
    srand(1023);

    // transition_prob
    float **transition_matrix = malloc(states * sizeof *transition_matrix);
    assert(transition_matrix);
    for (int i = 0; i < states; i++) {
        transition_matrix[i] = malloc(states * sizeof *transition_matrix[i]);
        assert(transition_matrix[i]);
    }
    for (i = 0; i < states; i++) {
        current_prob = 1;
        for (j = 0; j < states - 1; j++) {
            transition_matrix[i][j] = rand_prob(&current_prob);
            printf("%.4f ", transition_matrix[i][j]);
        }
        transition_matrix[i][j] = current_prob;
        printf("%.4f\n", current_prob);
    }

    // calc stat prob with Bayes rule, but here raise : loss of precision
    // ->  (0.9999 <= sum(P[i]) <= 1)
    float *P = malloc(states * sizeof *P);
    float denominator = 1;
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
        current_prob = 1;
        for (j = 0; j < observations - 1; j++) {
            printf("%.4f ", rand_prob(&current_prob));
        }
        printf("%.4f\n", current_prob);
    }

    // random obs
    printf("%d\n", observation_length);
    for (i = 0; i < observation_length; i++) {
        printf("%d\n", my_rand(0, observations - 1));
    }
    return 0;
}