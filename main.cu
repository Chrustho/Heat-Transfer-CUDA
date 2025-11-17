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

    
    int threadsInit = 1024;
    int blocksInit = (int)(((size/ sizeof(float)) + threadsInit - 1) / threadsInit);

    initTemperature<<<blocksInit,threadsInit>>>(deviceMatPrev, gridRows, gridCols, initialHotTemperature, nHotTopRows,nHotBottomRows);
    cudaDeviceSynchronize();

    //salva su file

    int dims[]={8,16,32};
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

            initTemperature<<<gridDim,blockDim>>>(deviceMatPrev, gridRows, gridCols, initialHotTemperature, nHotTopRows,nHotBottomRows);
            cudaDeviceSynchronize();

            // salva su file

            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start,0);

            for (size_t i = 0; i < nStep; i++)
            {
                // chiama kernel
            }

            cudaEventRecord(stop,0);
            cudaEventSynchronize(stop);

            float ms=0.0f;
            cudaEventElapsedTime(&ms, start, stop);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            
            // salvo su file
        }
    }



    return 0;
}
