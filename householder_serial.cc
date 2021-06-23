#include <string.h>

#include <cblas.h>

#include <stdio.h>

#include "cTimer.h"

#include "utils.cc"

int main(int argc, char * argv[]) {
	bool verbose = true;
	int i, j, k, l, lines;
	FILE * file;
	float ** matrixA, ** matrixQ, ** matrixR, ** matrixP;
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

	for (i = 0; i < lines; i++) {
		//Cauculate vec
		float * vec = new float[lines - i];

		for (j = i; j < lines; j++) {
			vec[j - i] = -matrixR[j][i];
		}

		float x_norm = cblas_snrm2(lines - i, vec, 1);

		if (vec[0] < 0) vec[0] = vec[0] + x_norm;
		else vec[0] = vec[0] - x_norm;

		float vec_norm = cblas_snrm2(lines - i, vec, 1);

		if (vec_norm > 0) {
			//Normalize vec
			for (j = 0; j < lines - i; j++) {
				vec[j] /= vec_norm;
			}

			//P
			matrixP = calculate_matrixP(vec, lines - i);

			//R
			float ** matrixR_part = matrix_multiply(matrixP, matrixR, 0, lines - i, 0, lines - i, i, lines, i, lines);
			for (k = 0; k < lines - i; k++) {
				for (l = 0; l < lines - i; l++) {
					matrixR[k + i][l + i] = matrixR_part[k][l];
				}
			}

			//Q
			float ** matrixQ_part = matrix_multiply(matrixP, matrixQ, 0, lines - i, 0, lines - i, i, lines, 0, lines);
			for (k = 0; k < lines - i; k++) {
				for (l = 0; l < lines; l++) {
					matrixQ[k + i][l] = matrixQ_part[k][l];
				}
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
