#include <cblas.h>

#include <stdio.h>

#include <algorithm>

#include <string.h>

#include <cmath>

#include "cTimer.h"

#include "utils.cc"

float ** hessenberg(float ** matrixA, int lines) {
	int i, j, k, l;
	float ** matrixA_part;

	for (i = 0; i < lines - 1; i++) {

		//Cauculate vec
		float * vec = new float[lines - i - 1];

		for (j = i; j < lines - 1; j++) {
			vec[j - i] = -matrixA[j + 1][i];
		}

		float x_norm = cblas_snrm2(lines - i - 1, vec, 1);

		if (vec[0] < 0) vec[0] = vec[0] + x_norm;
		else vec[0] = vec[0] - x_norm;

		float vec_norm = cblas_snrm2(lines - i - 1, vec, 1);

		if (vec_norm > 0) {
			//Normalize vec
			for (j = 0; j < lines - i - 1; j++) {
				vec[j] /= vec_norm;
			}

			//P
			float ** matrixP = calculate_matrixP(vec, lines - i - 1);
			//R
			matrixA_part = matrix_multiply(matrixP, matrixA, 0, lines - i - 1, 0, lines - i - 1, i + 1, lines, i, lines);

			for (k = 0; k < lines - i - 1; k++) {
				for (l = 0; l < lines - i; l++) {
					matrixA[k + i + 1][l + i] = matrixA_part[k][l];
				}
			}

			//Q
			matrixA_part = matrix_multiply(matrixA, matrixP, 0, lines, i + 1, lines, 0, lines - i - 1, 0, lines - i - 1);

			for (k = 0; k < lines; k++) {
				for (l = 0; l < lines - i - 1; l++) {
					matrixA[k][l + i + 1] = matrixA_part[k][l];
				}
			}

			delete[] vec;
			delete[] matrixP[0];
			delete[] matrixA_part[0];
		}
	}

	return matrixA;
}

float ** givens_update_shift(float ** matrixA, int lines) {
	int i, j, k;
	float ** matrixQ, ** matrixR;

//	int row1=rand()%lines;
//	int row2=rand()%lines;
//	if (row1==row2) row2=(row2+1)%lines;

	int row1 = lines - 2;
	int row2 = lines - 1;
	float shift = WilkinsonShift(matrixA[row1][row1], matrixA[row1][row2], matrixA[row2][row2]);

	//Initialize R=A Q=I
	matrixR = create_matrix(lines, lines);
	matrixQ = create_matrix(lines, lines);
	for (i = 0; i < lines; i++) {
		for (j = 0; j < lines; j++) {
			if (i == j) {
				matrixR[i][j] = matrixA[i][j] - shift;
				matrixQ[i][j] = 1;
			} else {
				matrixR[i][j] = matrixA[i][j];
				matrixQ[i][j] = 0;
			}
		}
	}

	for (j = 0; j < lines - 1; j++) {
		if (matrixR[j + 1][j] != 0) {
			matrixQ = givens_rotation(matrixQ, matrixR, j, j + 1, j, lines);
			matrixR = givens_rotation(matrixR, matrixR, j, j + 1, j, lines);
		}

	}
	float ** matrixQ_transpose = create_matrix(lines, lines);
	for (i = 0; i < lines; i++) {
		for (j = 0; j < lines; j++) {
			matrixQ_transpose[i][j] = matrixQ[j][i];
		}
	}
	matrixA = matrix_multiply(matrixR, matrixQ_transpose, 0, lines, 0, lines, 0, lines, 0, lines);
	for (i = 0; i < lines; i++) {
		matrixA[i][i] = matrixA[i][i] + shift;
	}

	return matrixA;
}

int main(int argc, char * argv[]) {
	bool verbose = true;
	float ** matrixA;
	int i, j, k, lines, rank;
	FILE * file;
	arg::cTimer timer;
	double time;
	int epoch = 0;
	float eps = 999;

	//using extra arguments from command line
	for (i = 0; i < argc; i++) {
		if (strcmp(argv[i], "-silent") == 0) verbose = false;
		if (strcmp(argv[i], "-file") == 0) {
			if ((file = fopen(argv[++i], "r")) == NULL) {
				printf("Cannot open file.\n");
				return 0;
			}
		}
	}
	//scanning file for number of lines
	fscanf(file, "%d,\n", & lines);
	matrixA = create_matrix(lines, lines);

	for (i = 0; i < lines; i++) {
		for (j = 0; j < lines; j++) {
			fscanf(file, "%f,", & matrixA[i][j]);
		}
	}
	if (verbose) {
		printf("Martix A.\n");
		for (i = 0; i < lines; i++) {
			for (j = 0; j < lines; j++) {
				printf("%f ", matrixA[i][j]);
			}
			printf("\n");
		}
	}
	//start time
	timer.CpuStart();

	matrixA = hessenberg(matrixA, lines);
	while (eps >= (float) lines / 100) {
		eps = 0;
		float * eigenvalues = new float[lines];
		for (i = 0; i < lines; i++) eigenvalues[i] = matrixA[i][i];

		std::sort(eigenvalues, eigenvalues + lines);

		for (i = 0; i < lines; i++) eps += std::abs(eigenvalues[i] - i - 1);

		matrixA = givens_update_shift(matrixA, lines);

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
		
		if (epoch > 1000) {
			printf("Exceeding max iteration: Early stopping\n");
			break;
		}
	}

	//end time
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

	return 0;
}
