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

#include "include/utility.h"
int main()
{
    size_t size= (size_t)gridRows*gridCols*sizeof(float);
    float *matPrev = (float *) malloc(size);
    float *matNext = (float *) malloc(size);

    float *deviceMatPrev= NULL;
    float *deviceMatNext= NULL;

    cudaMallocManaged((void**) &deviceMatPrev,size);
    cudaMallocManaged((void**) &deviceMatNext,size);

    // TODO: initMat

    // TODO: salvataggio su file iniziale

    // inizio timer CUDA

    for (size_t i=1; i <=nStep; i++)
    {
        // TODO  update region
        kernelHeatGlobal<<<,>>>();
        // TODO swap buffer
        swapBuffers(deviceMatNext, deviceMatNext);

    }

    // fine timer CUDA

    // stampa tempo

    // salva su file finale

    // delete


    



    return 0;
}
