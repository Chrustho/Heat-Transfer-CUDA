#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//void saveTemparature(const char* fileBaseName, 
//                     const char* fileExtension, 
//                     unsigned int step, 
//                     float* gridTemperature, 
//                     unsigned int nRows, 
//                     unsigned int nCols, 
//                     unsigned int fieldW)
//{
//    char* filename = NULL;
//    FILE* file = NULL;
//    int len = 0;
//
//    int step_len = snprintf(NULL, 0, "%u", step);
//    
//    len = strlen(fileBaseName) + strlen("_step_") + step_len + strlen(fileExtension) + 1;
//
//    filename = (char*)malloc(len);
//    if (filename == NULL) {
//        fprintf(stderr, "Errore: Impossibile allocare memoria per il nome del file.\n");
//        return;
//    }
//
//    snprintf(filename, len, "%s_step_%u%s", fileBaseName, step, fileExtension);
//
//    file = fopen(filename, "w");
//
//    if (file != NULL)
//    {
//        streamTemperature(file, step, gridTemperature, nRows, nCols, fieldW);
//        fclose(file);
//    }
//    else
//    {
//        fprintf(stderr, "Impossibile aprire il file %s!\n", filename);
//    }
//    
//    free(filename);
//}
//
//
//
//void swapBuffers(float*& prev, float*& curr) {
//  float *tmp = prev;
//  curr = prev;
//  curr = tmp;
//}
//
void printMatrix(float *mat, int w, int h) {
    for(int i = 0; i < w*h; i++) {
        printf("%f\t", mat[i]);
        if( i % w == 0 && i != 0)
            printf("\n");
    }
}
