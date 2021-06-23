#include <mpi.h>

#include <stdio.h>

#include <algorithm>

#include <string.h>

#include <cmath>

#include "cTimer.h"

#include "utils.cc"

#define MAX_PROCESSES 200

int main(int argc, char * argv[]) {
    bool verbose = true;
    float ** matrixA, ** matrixQ, ** matrixR;
    int start_dim[MAX_PROCESSES], end_dim[MAX_PROCESSES], displs[MAX_PROCESSES], send_counts[MAX_PROCESSES];
    arg::cTimer timer;
    double time;
    MPI_Status status;
    int i, j, k, q, lines, rank, size;
    FILE * file;
    float eps = 999;
    int epoch = 0;

    MPI_Init( & argc, & argv);
    MPI_Comm_rank(MPI_COMM_WORLD, & rank);
    MPI_Comm_size(MPI_COMM_WORLD, & size);

    if (rank != 0) verbose = false;
    if (rank == 0) {
        //using extra arguments from command line
        for (i = 0; i < argc; i++) {
            if (strcmp(argv[i], "-silent") == 0) verbose = false;
            if (strcmp(argv[i], "-file") == 0) {
                if ((file = fopen(argv[++i], "r")) == NULL) {
                    printf("Cannot open file.\n");
                    MPI_Finalize();
                    return 0;
                }
            }
        }

        //loading matrixA
        fscanf(file, "%d,\n", & lines);
    }
    MPI_Bcast( & lines, 1, MPI_INT, 0, MPI_COMM_WORLD);
    matrixA = create_matrix(lines, lines);
    matrixR = create_matrix(lines, lines);
    matrixQ = create_matrix(lines, lines);

    if (rank == 0) {
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

        //Cut rows or columns
        q = lines / size;

        for (k = 0; k < size; k++) {
            start_dim[k] = k * q;
            end_dim[k] = (k + 1) * q;
        }
        end_dim[size - 1] = lines;

        for (k = 0; k < size; k++) {
            send_counts[k] = (end_dim[k] - start_dim[k]) * lines;
            displs[k] = start_dim[k] * lines;
        }

        //start time
        timer.CpuStart();
    }
    MPI_Bcast( & start_dim[0], size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast( & end_dim[0], size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast( & send_counts[0], size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast( & displs[0], size, MPI_INT, 0, MPI_COMM_WORLD);

    int lines_part = end_dim[rank] - start_dim[rank];
    float ** matrixR_part = create_matrix(lines_part + 1, lines);
    float ** matrixQ_part = create_matrix(lines_part + 1, lines);

    while (eps >= (float) lines / 100) {
        if (rank == 0) {
            //Initialize R=A Q=I
            for (i = 0; i < lines; i++) {
                for (j = 0; j < lines; j++) {
                    matrixR[i][j] = matrixA[i][j];
                    if (i == j) matrixQ[i][j] = 1;
                    else matrixQ[i][j] = 0;
                }
            }
        }

        MPI_Scatterv( & matrixR[0][0], send_counts, displs, MPI_FLOAT, & matrixR_part[1][0], send_counts[rank], MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Scatterv( & matrixQ[0][0], send_counts, displs, MPI_FLOAT, & matrixQ_part[1][0], send_counts[rank], MPI_FLOAT, 0, MPI_COMM_WORLD);

        if (rank != size - 1) {
            MPI_Send( & matrixR_part[lines_part][0], lines, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD);
            MPI_Send( & matrixQ_part[lines_part][0], lines, MPI_FLOAT, rank + 1, 1, MPI_COMM_WORLD);
        }

        for (j = 0; j < end_dim[rank]; j++) {
            if (rank != size - 1) {
                MPI_Recv( & matrixR_part[lines_part][0], lines, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, & status);
                MPI_Recv( & matrixQ_part[lines_part][0], lines, MPI_FLOAT, rank + 1, 1, MPI_COMM_WORLD, & status);
            }

            if (lines_part > 1) {
                if (end_dim[rank] - 1 > j) {
                    matrixQ_part = givens_rotation(matrixQ_part, matrixR_part, lines_part - 1, lines_part, j, lines);
                    matrixR_part = givens_rotation(matrixR_part, matrixR_part, lines_part - 1, lines_part, j, lines);
                }

                if (rank != size - 1) {
                    MPI_Send( & matrixR_part[lines_part][0], lines, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD);
                    MPI_Send( & matrixQ_part[lines_part][0], lines, MPI_FLOAT, rank + 1, 1, MPI_COMM_WORLD);
                }
            }

            for (i = lines_part - 1; i > 1; i--) {
                if (i + start_dim[rank] - 1 > j) {
                    matrixQ_part = givens_rotation(matrixQ_part, matrixR_part, i - 1, i, j, lines);
                    matrixR_part = givens_rotation(matrixR_part, matrixR_part, i - 1, i, j, lines);
                }
            }

            if (start_dim[rank] > j) {
                if (rank != 0) {
                    MPI_Recv( & matrixR_part[0][0], lines, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, & status);
                    MPI_Recv( & matrixQ_part[0][0], lines, MPI_FLOAT, rank - 1, 1, MPI_COMM_WORLD, & status);
                }

                matrixQ_part = givens_rotation(matrixQ_part, matrixR_part, 0, 1, j, lines);
                matrixR_part = givens_rotation(matrixR_part, matrixR_part, 0, 1, j, lines);

                if (rank != 0) {
                    MPI_Send( & matrixR_part[0][0], lines, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD);
                    MPI_Send( & matrixQ_part[0][0], lines, MPI_FLOAT, rank - 1, 1, MPI_COMM_WORLD);
                }

                if (lines_part == 1) {
                    if (rank != size - 1) {
                        MPI_Send( & matrixR_part[1][0], lines, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD);
                        MPI_Send( & matrixQ_part[1][0], lines, MPI_FLOAT, rank + 1, 1, MPI_COMM_WORLD);
                    }
                }
            }
        }

        MPI_Gatherv( & matrixR_part[1][0], send_counts[rank], MPI_FLOAT, & matrixR[0][0], send_counts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Gatherv( & matrixQ_part[1][0], send_counts[rank], MPI_FLOAT, & matrixQ[0][0], send_counts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            float ** matrixQ_transpose = create_matrix(lines, lines);
            for (i = 0; i < lines; i++) {
                for (j = 0; j < lines; j++) {
                    matrixQ_transpose[i][j] = matrixQ[j][i];
                }
            }
            matrixA = matrix_multiply(matrixR, matrixQ_transpose, 0, lines, 0, lines, 0, lines, 0, lines);

            eps = 0;
            float * eigenvalues = new float[lines];
            for (i = 0; i < lines; i++) eigenvalues[i] = matrixA[i][i];
            std::sort(eigenvalues, eigenvalues + lines);
            for (i = 0; i < lines; i++) eps += std::abs(eigenvalues[i] - i - 1);

            if (verbose) {
                printf("\nMatrix A in %d round. Eps=%f.\n", epoch + 1, eps);
                for (i = 0; i < lines; i++) {
                    for (j = 0; j < lines; j++) {
                        printf("%f ", matrixA[i][j]);
                    }
                    printf("\n");
                }
            }
            epoch += 1;
        }

        MPI_Bcast( & eps, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Bcast( & epoch, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (epoch > 1000) {
            printf("Exceeding max iteration: Early stopping\n");
            break;
        }
    }

    //end time
    if (rank == 0) {
        time = timer.CpuStop().CpuSeconds();
        if (verbose) {
            printf("\nEig values:\n--------------------------------------\n");
            for (i = 0; i < lines; i++) {
                printf("%f\n", matrixA[i][i]);
            }
            printf("--------------------------------------\n");
        }
        printf("Time.\n%f\n", time);
        printf("Epoch.\n%d\n", epoch);
    }

    MPI_Finalize();
    return 0;
}
