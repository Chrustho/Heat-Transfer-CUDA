#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define MAX_TO_PRINT 8

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
void printMatrix2(float *mat, int w, int h) {
    for(int i = 0; i < w*h; i++) {
        printf("%f\t", mat[i]);
        if( (i+1) % w == 0 )
            printf("\n");
    }
}

void printMatrix(float *mat, int w, int h) {
    
    if (h == 0 || w == 0) {
        printf("Matrice vuota.\n");
        return;
    }

    int elementsToPrint = (w < MAX_TO_PRINT) ? w : MAX_TO_PRINT;
    bool puntiniPrintati = false;


    for (int i = 0; i < h; i++) {
        
        bool isTopRow = (i < nHotTopRows + 3);
        bool isBottomRow = (i >= h - nHotBottomRows - 3);

        if (isTopRow || isBottomRow) {
            
            for (int c = 0; c < elementsToPrint; c++) {
                int index = i * w + c;
                printf("%f\t", mat[index]);
            }
            printf("\n"); 

        } 
        else {
            if (!puntiniPrintati) {
                printf("...\n");
                puntiniPrintati = true;
            }
        }
    }

}
