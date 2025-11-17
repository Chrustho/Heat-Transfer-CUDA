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
#define G_MALLOC_ERROR -1
#define G_COPY_ERROR -2
#include "include/utility.h"
#include "include/init.cuh"
/**
* Metodo per l'allocazione di un'area di memoria sulla GPU
* Input: d, puntatore ad un puntatore di buffer di memoria
         qnt, quanto allocare
         isManaged, se l'area di memoria deve essere condivisa tra CPU e GPU
*/
void mallocGPU(float **d, int qnt, bool isManaged) {

    cudaError_t status;
    if(!isManaged)
        status = cudaMalloc((void **) d, (size_t) qnt);
    else 
        status = cudaMallocManaged((void**) d,qnt);

    if(status != cudaSuccess) {
        printf("Errore nell'allocazione di memoria sulla GPU: %s\n", cudaGetErrorString(status));
        exit(G_MALLOC_ERROR);
    }
}
/**
* Metodo per la copia di dati da/alla GPU
* Input: to, puntatore al buffer di destinazione
         from, puntatore al buffer di ricezione 
         size, intero che indica la dimensione dei buffer
         op, operazione scelta (cudaMemcpyDeviceToHost, cudaMemcpyHostToDevice)
*/
void copyGPU(float *to, float *from, int size, cudaMemcpyKind op) {
    cudaError_t status = cudaMemcpy(to, from, size, op);
    if(status != cudaSuccess) {
        printf("Errore nella copia di un elemento tra host e device %s\n", cudaGetErrorString(status));
        exit(G_COPY_ERROR);
    }
}
int main()
{
    size_t size= (size_t)DIMENSIONE*DIMENSIONE*sizeof(float);
    float *matPrev = (float *) malloc(size);
    float *matNext = (float *) malloc(size);

    float *deviceMatPrev= NULL;
    float *deviceMatNext= NULL;

    mallocGPU(&deviceMatPrev, size, true);
    mallocGPU(&deviceMatNext, size, true);

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
