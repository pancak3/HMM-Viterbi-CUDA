#include <stdio.h>
#include <stdlib.h>

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

    // start_prob
    for (i = 0; i < states - 1; i++) {
        printf("%.4f ", rand_prob(&current_prob));
    }
    printf("%.4f\n", current_prob);

    // transition_prob
    current_prob = 1;

    for (i = 0; i < states; i++) {
        current_prob = 1;
        for (j = 0; j < states - 1; j++) {
            printf("%.4f ", rand_prob(&current_prob));
        }
        printf("%.4f\n", current_prob);
    }

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
        printf("%d\n", my_rand(0, observations));
    }
    return 0;
}