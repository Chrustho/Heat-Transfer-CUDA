#include "include/init.cuh"
__global__ void initTemperature(float *mat, 
                                unsigned int rows, 
                                unsigned int cols, 
                                unsigned int initTemp, 
                                unsigned int topRows, 
                                unsigned int botRows){

    int col= blockIdx.x*blockDim.x+threadIdx.x;
    int row= blockIdx.y*blockDim.y+threadIdx.y;
    int idx= row*cols+col;

    if (row < rows && col < cols)
    {
        if (row>=topRows && row<=(rows-botRows-1))
        {
            mat[idx]=0.0f;
        }else{
            mat[idx]=(float) initTemp;
        }
        
    }
}
