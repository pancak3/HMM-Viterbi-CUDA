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
    int state_num, observation_num, rand_obs_num;
    scanf("%d %d %d", &state_num, &observation_num, &rand_obs_num);
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

    // random obs

    for (i = 0; i < rand_obs_num; i++) {
        printf("%d ", my_rand(0, observation_num));
    }
    printf("\n");
    return 0;
}