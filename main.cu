#define step 0
#define nStep 10000
#define gridRows 1 << 8
#define gridCols 1 << 12
#define nHotTopRows 2
#define nHotBottomRows 2
#define initialHotTemperature 20
#define fieldWidth 5

#define outFilePrefix "Temperature"
#define outFileExtension ".dat"
#define DIMENSIONE 8

#include "include/utility.h"
#include "init.cu"
int main()
{
    size_t size= (size_t)DIMENSIONE*DIMENSIONE*sizeof(float);
    float *matPrev = (float *) malloc(size);
    float *matNext = (float *) malloc(size);

    float *deviceMatPrev= NULL;
    float *deviceMatNext= NULL;

    cudaMallocManaged((void**) &deviceMatPrev,size);
    cudaMallocManaged((void**) &deviceMatNext,size);

    // TODO: initMat
    dim3 blockDim(DIMENSIONE, DIMENSIONE);
    dim3 gridDim((DIMENSIONE + DIMENSIONE - 1) / DIMENSIONE, 
                 (DIMENSIONE + DIMENSIONE - 1) / DIMENSIONE);
    initTemperature<<<gridDim,blockDim>>>(deviceMatPrev, DIMENSIONE, DIMENSIONE, 20, 2,2);
    cudaDeviceSynchronize();
    printf("Accesso dalla CPU all'elemento [0][0]: %f\n", deviceMatPrev[0]);
    printMatrix(deviceMatPrev, DIMENSIONE, DIMENSIONE);

    // TODO: salvataggio su file iniziale

    // inizio timer CUDA

//    for (size_t i=1; i <=nStep; i++)
//    {
//        // TODO  update region
//        kernelHeatGlobal<<<,>>>();
//        // TODO swap buffer
//        swapBuffers(deviceMatNext, deviceMatNext);
//
//    }
//
    // fine timer CUDA

    // stampa tempo

    // salva su file finale

    // delete


    



    return 0;
}
