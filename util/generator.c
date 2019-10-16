#include <stdio.h>
#include <stdlib.h>

float rand_prob(float *high);

float rand_prob(float *high) {

    int max = *high * 10000;
    int min = 0;
    if (max == 0) {
        return 0;
    }
    float t = (float) ((rand() % (max - min + 1) + min)) / 10000;
    *high -= t;

    return t;

}

int main() {
    int i, j;
    int state_num, observation_num;
    scanf("%d %d", &state_num, &observation_num);
    printf("%d %d\n", state_num, observation_num);
    float current_prob = 1;
    srand(1023);

    // start_prob
    for (i = 0; i < state_num - 1; i++) {
        printf("%.4f ", rand_prob(&current_prob));
    }
    printf("%.4f\n", current_prob);

    // transition_prob
    current_prob = 1;

    for (i = 0; i < state_num; i++) {
        current_prob = 1;
        for (j = 0; j < state_num - 1; j++) {
            printf("%.4f ", rand_prob(&current_prob));
        }
        printf("%.4f\n", current_prob);
    }

    // emission_prob
    for (i = 0; i < state_num; i++) {
        current_prob = 1;
        for (j = 0; j < observation_num - 1; j++) {
            printf("%.4f ", rand_prob(&current_prob));
        }
        printf("%.4f\n", current_prob);
    }
    return 0;
}