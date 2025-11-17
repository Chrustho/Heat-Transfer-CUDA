__global__ void initTemperature(float *mat, unsigned int gridRows, unsigned int gridCols, unsigned int initTemp, unsigned int topRows, unsigned botRows){

    int tx= threadIdx.x;
    int idx = blockIdx.x*blockDim.x+tx;
    int size= gridRows*gridCols;


    if (idx < size)
    {
        if (idx>gridCols*topRows && idx<size-gridCols*botRows)
        {
            mat[idx]=0.0f;
        }else{
            mat[idx]=20.0f;
        }
    }
    
    
    
    
}