#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "salva.h"

void saveTemparature(const char* file_base_name, const char* file_extension, unsigned int step, double* gridTemperature, unsigned int nRows, unsigned int nCols, unsigned int fieldW)
{
    char* filename = NULL;
    FILE* file = NULL;
    int len = 0;

    int step_len = snprintf(NULL, 0, "%u", step);
    
    len = strlen(file_base_name) + strlen("_step_") + step_len + strlen(file_extension) + 1;

    filename = (char*)malloc(len);
    if (filename == NULL) {
        fprintf(stderr, "Errore: Impossibile allocare memoria per il nome del file.\n");
        return;
    }

    snprintf(filename, len, "%s_step_%u%s", file_base_name, step, file_extension);

    file = fopen(filename, "w");

    if (file != NULL)
    {
        streamTemperature(file, step, gridTemperature, nRows, nCols, fieldW);
        fclose(file);
    }
    else
    {
        fprintf(stderr, "Impossibile aprire il file %s!\n", filename);
    }
    
    free(filename);
}



void swapBuffers(float*& prev, float*& curr) {
  float *tmp = prev;
  curr = prev;
  curr = tmp;
}
