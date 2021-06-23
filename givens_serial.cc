#include <string.h>

#include <stdio.h>

#include "cTimer.h"

#include "utils.cc"

int main(int argc, char * argv[]) {
    bool verbose = true;
    int i, j, lines;
    FILE * file;
    float ** matrixA, ** matrixQ, ** matrixR;
    arg::cTimer timer;
    double time;

    for (i = 0; i < argc; i++) {
        if (strcmp(argv[i], "-silent") == 0) verbose = false;
        if (strcmp(argv[i], "-file") == 0) {
            if ((file = fopen(argv[++i], "r")) == NULL) {
                printf("Cannot open file.\n");
                return 0;
            }
        }
    }

    //loading matrixA
    fscanf(file, "%d,\n", & lines);
    matrixA = create_matrix(lines, lines);
    for (i = 0; i < lines; i++) {
        for (j = 0; j < lines; j++) {
            fscanf(file, "%f,", & matrixA[i][j]);
        }
    }

    if (verbose) {
        printf("Matrix A.\n");
        for (i = 0; i < lines; i++) {
            for (j = 0; j < lines; j++) {
                printf("%f ", matrixA[i][j]);
            }
            printf("\n");
        }
    }

    //Initialize R=A Q=I
    matrixR = create_matrix(lines, lines);
    matrixQ = create_matrix(lines, lines);
    for (i = 0; i < lines; i++) {
        for (j = 0; j < lines; j++) {
            matrixR[i][j] = matrixA[i][j];
            if (i == j) matrixQ[i][j] = 1;
            else matrixQ[i][j] = 0;
        }
    }

    //start time
    timer.CpuStart();

    for (j = 0; j < lines; j++) {
        for (i = lines - 1; i > j; i--) {
            if (matrixR[i][j] != 0) {
                matrixQ = givens_rotation(matrixQ, matrixR, i - 1, i, j, lines);
                matrixR = givens_rotation(matrixR, matrixR, i - 1, i, j, lines);
            }
        }
    }

    //end time
    time = timer.CpuStop().CpuSeconds();

    if (verbose) {
        printf("\nSolution is:\n");
        printf("Matrix Q.\n");
        for (i = 0; i < lines; i++) {
            for (j = 0; j < lines; j++) {
                // matrixQ is actually transpose of what we want
                printf("%f ", matrixQ[j][i]);
            }
            printf("\n");
        }
        printf("Matrix R.\n");
        for (i = 0; i < lines; i++) {
            for (j = 0; j < lines; j++) {
                printf("%f ", matrixR[i][j]);
            }
            printf("\n");
        }
    }
    printf("Time: %f\n", time);

    return 0;
}
