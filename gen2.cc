// Generate symmetric matrix with given eigenvalues 1,2,...,n, where n is the matrix size.

#include <iostream>

#include <string.h>

#include <stdbool.h>

#include <stdio.h>

#include <stdlib.h>

#include <cblas.h>

#include "utils.cc"

int main(int argc, char * argv[]) {
    int lines = atoi(argv[1]);

    float ** matrixA = create_matrix(lines, lines);
    for (int i = 0; i < lines; i++) {
        matrixA[i][i] = i + 1;
    }

    for (int epoch = 0; epoch < 20; epoch++) {
        //Generate random unit vector
        float * vec = new float[lines];
        for (int i = 0; i < lines; i++) {
            vec[i] = rand() % lines;
        }
        float vec_norm = cblas_snrm2(lines, vec, 1);
        for (int i = 0; i < lines; i++) {
            vec[i] /= vec_norm;
        }

        float ** matrixP = calculate_matrixP(vec, lines);

        matrixA = matrix_multiply(matrixP, matrixA, 0, lines, 0, lines, 0, lines, 0, lines);
        matrixA = matrix_multiply(matrixA, matrixP, 0, lines, 0, lines, 0, lines, 0, lines);
    }

    printf("%d,\n", lines);
    for (int i = 0; i < lines; i++) {
        for (int j = 0; j < lines; j++) {
            printf("%f,", matrixA[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}
