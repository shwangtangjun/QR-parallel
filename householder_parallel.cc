#include "mpi.h"

#include <cblas.h>

#include <string.h>

#include <stdio.h>

#include "cTimer.h"

#include "utils.cc"

#define MAX_PROCESSES 200

int main(int argc, char * argv[]) {
    bool verbose = true;
    int i, j, k, l, rank, size, lines;
    int q;
    FILE * file;
    float ** matrixA, ** matrixQ, ** matrixR, ** matrixP;
    int start_dim[MAX_PROCESSES], end_dim[MAX_PROCESSES];
    int displs1[MAX_PROCESSES], send_counts1[MAX_PROCESSES], displs2[MAX_PROCESSES], send_counts2[MAX_PROCESSES];
    arg::cTimer timer;
    double time;

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

        //Initialize R=A Q=I
        for (i = 0; i < lines; i++) {
            for (j = 0; j < lines; j++) {
                matrixR[i][j] = matrixA[i][j];
                if (i == j) matrixQ[i][j] = 1;
                else matrixQ[i][j] = 0;
            }
        }
        //start time
        timer.CpuStart();
    }

    for (i = 0; i < lines; i++) {
        float * vec = new float[lines - i];
        float vec_norm;
        matrixP = create_matrix(lines - i, lines - i);

        if (rank == 0) {
            q = (lines - i) / size;

            for (k = 0; k < size; k++) {
                start_dim[k] = k * q;
                end_dim[k] = (k + 1) * q;
            }
            end_dim[size - 1] = lines - i;

            for (k = 0; k < size; k++) {
                send_counts1[k] = (end_dim[k] - start_dim[k]) * (lines - i);
                displs1[k] = start_dim[k] * (lines - i);
                send_counts2[k] = (end_dim[k] - start_dim[k]) * lines;
                displs2[k] = start_dim[k] * lines;
            }

            //Cauculate vec
            for (j = i; j < lines; j++) {
                vec[j - i] = -matrixR[j][i];
            }

            float x_norm = cblas_snrm2(lines - i, vec, 1);

            if (vec[0] < 0) vec[0] = vec[0] + x_norm;
            else vec[0] = vec[0] - x_norm;
            vec_norm = cblas_snrm2(lines - i, vec, 1);

            if (vec_norm > 0) {
                //Normalize vec
                for (j = 0; j < lines - i; j++) {
                    vec[j] /= vec_norm;
                }

                //P
                matrixP = calculate_matrixP(vec, lines - i);
            }
        }

        MPI_Bcast( & start_dim[0], size, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast( & end_dim[0], size, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast( & send_counts1[0], size, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast( & displs1[0], size, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast( & send_counts2[0], size, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast( & displs2[0], size, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast( & vec_norm, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

        int lines_part = end_dim[rank] - start_dim[rank];
        float ** matrixP_part = create_matrix(lines_part, lines - i);

        MPI_Scatterv( & matrixP[0][0], send_counts1, displs1, MPI_FLOAT, & matrixP_part[0][0], send_counts1[rank], MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Bcast( & matrixQ[0][0], lines * lines, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Bcast( & matrixR[0][0], lines * lines, MPI_FLOAT, 0, MPI_COMM_WORLD);

        if (vec_norm > 0) {
            //R
            float ** matrixR_part = create_matrix(lines - i, lines - i);
            float ** matrixR_part_part = matrix_multiply(matrixP_part, matrixR, 0, lines_part, 0, lines - i, i, lines, i, lines);

            MPI_Gatherv( & matrixR_part_part[0][0], send_counts1[rank], MPI_FLOAT, & matrixR_part[0][0], send_counts1, displs1, MPI_FLOAT, 0, MPI_COMM_WORLD);

            if (rank == 0) {
                for (k = 0; k < lines - i; k++) {
                    for (l = 0; l < lines - i; l++) {
                        matrixR[k + i][l + i] = matrixR_part[k][l];
                    }
                }
            }

            //Q
            float ** matrixQ_part = create_matrix(lines - i, lines);
            float ** matrixQ_part_part = matrix_multiply(matrixP_part, matrixQ, 0, lines_part, 0, lines - i, i, lines, 0, lines);

            MPI_Gatherv( & matrixQ_part_part[0][0], send_counts2[rank], MPI_FLOAT, & matrixQ_part[0][0], send_counts2, displs2, MPI_FLOAT, 0, MPI_COMM_WORLD);

            if (rank == 0) {
                for (k = 0; k < lines - i; k++) {
                    for (l = 0; l < lines; l++) {
                        matrixQ[k + i][l] = matrixQ_part[k][l];
                    }
                }
            }
        }
    }

    if (rank == 0) {
        //end time
        time = timer.CpuStop().CpuSeconds();
        if (verbose) {
            printf("\nSolution is:\n");
            printf("Matrix Q.\n");
            for (i = 0; i < lines; i++) {
                for (j = 0; j < lines; j++) {
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
    }

    MPI_Finalize();
    return 0;
}
