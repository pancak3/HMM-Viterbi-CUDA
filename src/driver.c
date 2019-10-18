/**
 * Driver code to read serialised input data for the Viterbi algorithm.
 * Requires an implementation of the Viterbi algorithm to produce output.
 *
 * COMP90025 Parallel and Multicore Computing
 * Qifan Deng (qifand@student.unimelb.edu.au)
 * Alan Ung (alanu@student.unimelb.edu.au)
 * Semester 2, 2019
 */

#include "../headers/driver.h"
#include "../headers/viterbi_sequential.h"
#include <math.h>


int main(int argc, char const **argv) {
    // read in the number of possible hidden states and emissions
    int states, emissions, observations_length;
    if (scanf("%d %d", &states, &emissions) != 2) {
#ifdef DEBUG
        fprintf(stderr, "Cannot read # possible states and emissions.\n");
#endif // DEBUG
        exit(EXIT_FAILURE);
    }
    double *init_probabilities = read_init_probabilities(stdin, states);
    double **transition_matrix = read_transition_matrix(stdin, states);
    double **emission_table = read_emission_table(stdin, states, emissions);
    scanf("%d", &observations_length);
    int *observation_table = read_observation(stdin, observations_length);

    for (int i = 0; i < states; i++) {
        init_probabilities[i] = log(init_probabilities[i]);
        for (int j = 0; j < states; j++)
            transition_matrix[i][j] = log(transition_matrix[i][j]);
        for (int j = 0; j < emissions; j++)
            emission_table[i][j] = log(emission_table[i][j]);
    }


#ifdef DEBUG
    printf("States: %d, Emissions: %d, Observation_length: %d\n", states,
           emissions, observations_length);
    printf("[INIT PROBABILITIES]\n");
    for (int i = 0; i < states; i++)
        printf("%.4lf ", init_probabilities[i]);
    printf("\n");
    printf("[EMISSION PROBABILITIES]\n");
    for (int i = 0; i < states; i++) {
        for (int j = 0; j < emissions; j++)
            printf("%.4lf ", emission_table[i][j]);
        printf("\n");
    }
    printf("[TRANSITION MATRIX]\n");
    for (int i = 0; i < states; i++) {
        for (int j = 0; j < states; j++)
            printf("%.4lf ", transition_matrix[i][j]);
        printf("\n");
    }

    printf("[OBSERVATION TABLE]\n");
    for (int i = 0; i < observations_length; i++) {
        printf("%d ", observation_table[i]);
    }
    printf("\n");
#endif // DEBUG

    int *optimal_path = viterbi_sequential(states, emissions, observation_table,
                                           observations_length,
                                           init_probabilities,
                                           (const double **) transition_matrix,
                                           (const double **) emission_table);

#ifdef DEBUG
    printf("[OPTIMAL PATH]\n");
    for (int i = 0; i < observations_length; i++) {
        printf("%d ", optimal_path[i]);
    }
    putchar('\n');
#endif

//    free(init_probabilities);
//    free_2D_memory(transition_matrix, states);
//    free_2D_memory(emission_table, states);
}

/**
 * Allocate and read an array of initial probabilities from a specified file.
 * Calling function must free() memory allocated.
 * @param f File from which to read the probabilities
 * @param states Number of possible states
 * @return Pointer to array of initial probabilities
 */
double *read_init_probabilities(FILE *f, int states) {
    // allocate memory
    double *init_probs = malloc(states * sizeof *init_probs);
    assert(init_probs);

    // read in values
    for (int i = 0; i < states; i++)
        if (!fscanf(f, "%lf", init_probs + i))
            return NULL;

    return init_probs;
}

/**
 * Allocate and read an array of observations table from a specified file.
 * Calling function must free() memory allocated.
 * @param f File from which to read the observations table
 * @param observations Number of observations table
 * @return Pointer to array of observations table
 */
int *read_observation(FILE *f, int observations) {
    // allocate memory
    int *observations_table = malloc(observations * sizeof *observations_table);
    assert(observations_table);

    // read in values
    for (int i = 0; i < observations; i++)
        if (!fscanf(f, "%d", observations_table + i))
            return NULL;

    return observations_table;
}

/**
 * Allocate memory and read a transition matrix from a specified file.
 * Calling function must call free_2D_memory() to free memory.
 * @param f File from which to read the transition matrix
 * @param states Number of possible states (matrix width)
 * @return Pointer to transition matrix
 */
double **read_transition_matrix(FILE *f, int states) {
    // allocate memory
    double **transition_matrix = malloc(states * sizeof *transition_matrix);
    assert(transition_matrix);
    for (int i = 0; i < states; i++) {
        transition_matrix[i] = malloc(states * sizeof *transition_matrix[i]);
        assert(transition_matrix[i]);
    }

    // read in values
    for (int i = 0; i < states; i++)
        for (int j = 0; j < states; j++)
            if (!fscanf(f, "%lf", transition_matrix[i] + j))
                return NULL;

    return transition_matrix;
}

/**
 * Allocate memory and read an emission probability table from a specified file.
 * Calling function must call free_2D_memory() to free memory.
 * @param f File from which to read the emission probability table
 * @param states Number of possible states (table height)
 * @param emissions Number of possible emissions (table width)
 * @return Pointer to emission probability table
 */
double **read_emission_table(FILE *f, int states, int emissions) {
    // allocate memory
    double **table = malloc(states * sizeof *table);
    assert(table);
    for (int i = 0; i < states; i++) {
        table[i] = malloc(emissions * sizeof *table[i]);
        assert(table[i]);
    }

    // read in values
    for (int i = 0; i < states; i++)
        for (int j = 0; j < emissions; j++)
            if (!fscanf(f, "%lf", table[i] + j))
                return NULL;

    return table;
}

/**
 * Free memory allocated by read_transition_matrix() or read_emission_table().
 * @param table Pointer to table
 * @param states Number of rows
 */
void free_2D_memory(double **table, int rows) {
    if (!table)
        return;
    for (int i = 0; i < rows; i++)
        free(table[i]);
    free(table);
}