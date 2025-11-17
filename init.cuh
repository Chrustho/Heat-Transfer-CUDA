__global__ void initTemperature(float *mat, unsigned int gridRows, unsigned int gridCols, unsigned int initTemp, unsigned int topRows, unsigned int botRows){

    int col= blockIdx.x*blockDim.x+threadIdx.x;
    int row= blockIdx.y*blockDim.y*threadIdx.y;
    int idx= row*gridCols+col;

    if (row < gridRows && col < gridCols)
    {
        if (row<topRows || row>= (gridRows-botRows))
        {
            mat[idx]=(float) initTemp;
        }else{
            mat[idx]=0.0f;
        }
        
    }

    
    
    
}