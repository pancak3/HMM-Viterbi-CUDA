#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define STATE_NUM 2
#define OBSERVATION_NUM 2

int a = 2;

double rand_prob(double low, double high);

double rand_prob(double low, double high) {

    return 0.1;
}

int main() {
    srand(time(NULL));
    printf("%d", a);
    return 0;
}