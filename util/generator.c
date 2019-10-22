#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>

#define MAX_SIZE 10000
#define FACTOR 5

double min_prob = 1 / MAX_SIZE;
int scale = MAX_SIZE * FACTOR;

double rand_prob(double *high);

int my_rand(int min, int max);

int pancake_rand();


double prob_sum = 1.0;

double rand_prob(double *high) {
    int max = *high * scale;
    int min = 1;
    if (max == 0) {
        return min_prob;
    }
    double t;

    t = (double) (my_rand(min, max)) / scale;
    t *= (double) (my_rand(min, max)) / scale;

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

    double current_prob;
    srand(clock());

    // start prob
    current_prob = prob_sum;
    for (int i = 0; i < states - 1; i++) {
        printf("%.10f ", rand_prob(&current_prob));
    }
    printf("%.10f\n", current_prob);

    // transition_prob
    for (i = 0; i < states; i++) {
        current_prob = prob_sum;
        for (j = 0; j < states - 1; j++) {
            printf("%.10f ", rand_prob(&current_prob));
        }
        printf("%.10f\n", current_prob);
    }

    // emission_prob
    for (i = 0; i < states; i++) {
        current_prob = 1;
        for (j = 0; j < observations - 1; j++) {
            printf("%.10f ", rand_prob(&current_prob));
        }
        printf("%.10f\n", current_prob);
    }

    // random obs
    printf("%d\n", observation_length);
    for (i = 0; i < observation_length; i++) {
        printf("%d\n", my_rand(0, observations - 1));
    }
    putchar('\n');
    return 0;
}