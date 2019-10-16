#include <stdio.h>
#include <stdlib.h>
#include <time.h>

float rand_prob(float *high);

float rand_prob(float *high) {
    int max = *high * 10000;
    int min = 1;

    float t = (float) ((rand() % (max - min + 1) + min)) / 10000;
    *high -= t;

    return t;

}

int main() {
    int i, j;
    int state_num, observation_num;
    scanf("%d %d", &state_num, &observation_num);

    float current_prob = 1;
    srand(clock());

    float *start_prob = malloc(state_num * sizeof *start_prob);
    float *transition_prob = malloc(state_num * state_num * sizeof *transition_prob);
    float *emission_prob = malloc(state_num * observation_num * sizeof *emission_prob);

    // start_prob
    for (i = 0; i < state_num - 1; i++) {
        start_prob[i] = rand_prob(&current_prob);
    }
    start_prob[i] = current_prob;

    // transition_prob
    current_prob = 1;

    for (i = 0; i < state_num; i++) {
        current_prob = 1;
        for (j = 0; j < state_num - 1; j++) {
            transition_prob[i * state_num + j] = rand_prob(&current_prob);
        }
        transition_prob[i * state_num + j] = current_prob;
    }

    // emission_prob
    for (i = 0; i < state_num; i++) {
        current_prob = 1;
        for (j = 0; j < observation_num - 1; j++) {
            emission_prob[i * observation_num + j] = rand_prob(&current_prob);
        }
        emission_prob[i * observation_num + j] = current_prob;
    }

    // output
    printf("%d %d\n", state_num, observation_num);

    for (i = 0; i < state_num; i++) {
        printf("%.4f ", start_prob[i]);
    }
    printf("\n");

    for (i = 0; i < state_num; i++) {
        for (j = 0; j < state_num; j++) {
            printf("%.4f ", transition_prob[i * state_num + j]);
        }
        printf("\n");
    }

    for (i = 0; i < state_num; i++) {
        for (j = 0; j < observation_num; j++) {
            printf("%.4f ", emission_prob[i * observation_num + j]);
        }
        printf("\n");

    }
    return 0;
}