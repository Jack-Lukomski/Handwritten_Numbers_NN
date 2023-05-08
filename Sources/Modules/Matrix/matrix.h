#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifndef MATRIX_H
#define MATRIX_H

typedef struct {
    uint8_t rows;
    uint8_t cols;
    double * matrixData;
} Matrix;

Matrix * xCreateMatrix(uint8_t rows, uint8_t cols, double * newData);
Matrix * xDotProduct(Matrix * matrix1, Matrix * matrix2);
Matrix * xMatrixAdd(Matrix * matrix1, Matrix * matrix2);
static Matrix * xCreateEmptyMatrix(uint8_t rows, uint8_t cols);
static double * xGetRow(Matrix * m, uint8_t rowNum);
static double * xGetCol(Matrix * m, uint8_t colNum);

void vPrintMatrix(Matrix * m);

#endif