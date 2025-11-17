#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
void streamTemperature(FILE* file, unsigned int step, double* gridTemperature, unsigned int nRows, unsigned int nCols, unsigned int fieldW)
{
    for (unsigned int i = 0; i < nRows; i++)
    {
        for (unsigned int j = 0; j < nCols; j++)
        {
            fprintf(file, "%*f ", (int)fieldW, gridTemperature[i*nCols+j]);
        }
        fprintf(file, "\n");
    }
    fprintf(file, "\n");
}

void printTemperature(unsigned int step, double* gridTemperature, unsigned int nRows, unsigned int nCols, unsigned int fieldW)
{
    streamTemperature(stdout, step, gridTemperature, nRows, nCols, fieldW);
}

