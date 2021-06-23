#include <iostream>

#include <stdio.h>

#include <stdlib.h>

#include <cblas.h>

#include <math.h>

// Generate contiguous 2d array
float ** create_matrix(int numrows, int numcols) {
    float * buffer = new float[numrows * numcols];
    float ** data = new float * [numrows];
    for (int i = 0; i < numrows; i++) data[i] = buffer + i * numcols;

    return data;
}

// Calculate matrixX * matrixY, the other parameters control which rows or columns to choose
float ** matrix_multiply(float ** matrixX, float ** matrixY, int X_row_start, int X_row_end, int X_col_start, int X_col_end, int Y_row_start, int Y_row_end, int Y_col_start, int Y_col_end) {
    int X_row = X_row_end - X_row_start;
    int X_col = X_col_end - X_col_start;
    int Y_row = Y_row_end - Y_row_start;
    int Y_col = Y_col_end - Y_col_start;

    if (X_col != Y_row) {
        printf("Error. Matrix multiplication must have X_col = Y_row!");
        return 0;
    }
    float * x = new float[X_row * X_col];
    float * y = new float[Y_row * Y_col];
    float * z = new float[X_row * Y_col];

    for (int i = 0; i < X_row; i++) {
        for (int j = 0; j < X_col; j++) {
            x[i * X_col + j] = matrixX[X_row_start + i][X_col_start + j];
        }
    }
    for (int i = 0; i < Y_row; i++) {
        for (int j = 0; j < Y_col; j++) {
            y[i * Y_col + j] = matrixY[Y_row_start + i][Y_col_start + j];
        }
    }

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, X_row, Y_col, X_col, 1, x, X_col, y, Y_col, 0, z, Y_col);

    float ** matrixZ = create_matrix(X_row, Y_col);
    for (int i = 0; i < X_row; i++) {
        for (int j = 0; j < Y_col; j++) {
            matrixZ[i][j] = z[i * Y_col + j];
        }
    }

    delete[] x;
    delete[] y;
    delete[] z;

    return matrixZ;
}

//Symmetric rank 1 update P=I-2*vec*vec^T
float ** calculate_matrixP(float * vec, int dim) {
    float * identity = new float[dim * dim];
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            if (i == j) identity[i * dim + j] = 1;
            else identity[i * dim + j] = 0;
        }
    }

    cblas_ssyr(CblasRowMajor, CblasLower, dim, -2, vec, 1, identity, dim);
    float ** matrixP = create_matrix(dim, dim);
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            if (i >= j) matrixP[i][j] = identity[i * dim + j];
            else matrixP[i][j] = identity[j * dim + i];

        }
    }

    return matrixP;
}

float ** givens_rotation(float ** matrix, float ** matrixA, int row1, int row2, int col, int lines) {
    //matrix: the matrix to rotate
    //matrixA: the matrix that provide parameters

    float * rotation = new float[4];
    float * matrix_part = new float[2 * lines];
    float * matrix_part_new = new float[2 * lines];

    float t = matrixA[row1][col] / matrixA[row2][col];
    float s = 1 / sqrt(1 + t * t);
    float c = s * t;

    rotation[0] = c;
    rotation[1] = s;
    rotation[2] = -s;
    rotation[3] = c;

    for (int j = 0; j < lines; j++) {
        matrix_part[j] = matrix[row1][j];
        matrix_part[lines + j] = matrix[row2][j];
    }

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 2, lines, 2, 1, rotation, 2, matrix_part, lines, 0, matrix_part_new, lines);

    for (int j = 0; j < lines; j++) {
        matrix[row1][j] = matrix_part_new[j];
        matrix[row2][j] = matrix_part_new[lines + j];
    }

    delete[] rotation;
    delete[] matrix_part;
    delete[] matrix_part_new;

    return matrix;
}

//Calculate Wilkinson's shift for symmetric matrices
float WilkinsonShift(float a, float b, float c) {
	float d= (a-c)/2;

	if (d>0) return c- b*b/(d+sqrt(d*d+b*b));
	else return c- b*b/(d-sqrt(d*d+b*b));

}
