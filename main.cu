#define step 0
#define nStep 10000
#define gridRows (1 << 8)
#define gridCols (1 << 12)
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
#include "include/kernel.cuh"
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
/**
* Metodo per l'esecuzione del kernel e la valutazione in ms del tempo di esecuzione.
* Questo esegue una simulazione parallela per un numero di passi pari a nStep.
*/
void runKernel(dim3 blockDim, dim3 gridDim, int dim1, int dim2, float *matNext, float *matPrev) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start,0);
        size_t sharedMemSize = ((dim1 +1) * dim2) * sizeof(float);
        size_t sharemMemSize_wH = (dim1 + 2) * (dim2 + 2) * sizeof(float);

        for (size_t i = 0; i < nStep; i++)
        {
            //updateTiledOptimized<<<gridDim,blockDim,sharedMemSize>>>(matNext,matPrev,gridCols,gridRows,nHotBottomRows,nHotTopRows,dim1,dim2);
            updateTiledOptimized<<<gridDim,blockDim, sharedMemSize>>>(matNext,matPrev,gridCols,gridRows,nHotTopRows,nHotBottomRows, dim1, dim2);
            //tiled_wH_corrected<<<gridDim,blockDim,sharemMemSize_wH>>>(matNext,matPrev,gridCols,gridRows,nHotTopRows,nHotBottomRows);
            cudaDeviceSynchronize();
            float *temp = matPrev;
            matPrev = matNext;
            matNext = temp;
        }

        cudaEventRecord(stop,0);
        cudaEventSynchronize(stop);

        float ms=0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        printf("Tempo esecuzione blocco %d x %d: %f ms\n", dim1,dim2,ms);
}

int main()
{
    size_t size= (size_t)gridRows*gridCols*sizeof(float);
    float *matPrev = (float *) malloc(size);
    float *matNext = (float *) malloc(size);

    float *deviceMatPrev= NULL;
    float *deviceMatNext= NULL;

    mallocGPU(&deviceMatPrev, size, true);
    mallocGPU(&deviceMatNext, size, true);
    //int threadsInit = 1024;
    //int blocksInit = (int)(((size/ sizeof(float)) + threadsInit - 1) / threadsInit);
    //initTemperature<<<blocksInit,threadsInit>>>(deviceMatPrev, gridRows, gridCols, initialHotTemperature, nHotTopRows,nHotBottomRows);
    //cudaDeviceSynchronize();
    //salva su file

    int dims[]={8,16,32};
    int numRun=4;
    for (size_t i = 0; i < 3; i++)
    {
        for (size_t j = 0; j < 3; j++)
        {
            int dim1 = dims[i]; 
            int dim2 = dims[j]; 

            
            if (dim1 * dim2 > 1024)
            {
                continue; 
            }

            dim3 blockDim(dim1, dim2);
            dim3 gridDim((gridCols + dim1 - 1) / dim1, 
                         (gridRows + dim2 - 1) / dim2);
            for (size_t k = 0; k < numRun; k++)
            {
                initTemperature<<<gridDim,blockDim>>>(deviceMatPrev, gridRows, gridCols, initialHotTemperature, nHotTopRows,nHotBottomRows);
                initTemperature<<<gridDim,blockDim>>>(deviceMatNext, gridRows, gridCols, initialHotTemperature, nHotTopRows,nHotBottomRows);
                cudaDeviceSynchronize();

                if (i == 2 && j == 2) {
                printf("Matrice inizializzata:\n");
                printMatrix(deviceMatPrev, gridCols, gridRows); 
                }

                runKernel(blockDim,gridDim,dim1,dim2,deviceMatNext,deviceMatPrev);
                cudaDeviceSynchronize();


                if (i==2 && j==2)
                {
                printMatrix(deviceMatPrev,gridCols,gridRows);
                }
            }
            

        }
    }

    return 0;
}
