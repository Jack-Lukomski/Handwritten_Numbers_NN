#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifndef MATRIX_H
#define MATRIX_H

typedef struct {
    uint16_t rows;
    uint16_t cols;
    double * matrixData;
} Matrix;

Matrix * xCreateMatrix(uint16_t rows, uint16_t cols, double * newData);
Matrix * xDotProduct(Matrix * matrix1, Matrix * matrix2);
Matrix * xMatrixAdd(Matrix * matrix1, Matrix * matrix2);
Matrix * xMatrixSubtract(Matrix * matrix1, Matrix * matrix2);
Matrix * xMatrixSquareEachElement(Matrix * m);
static Matrix * xCreateEmptyMatrix(uint16_t rows, uint16_t cols);
static double * xGetRow(Matrix * m, uint16_t rowNum);
static double * xGetCol(Matrix * m, uint16_t colNum);

void vPrintMatrix(Matrix * m);

#endif