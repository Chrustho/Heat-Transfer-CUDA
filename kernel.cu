#include "include/kernel.cuh"

__global__ void updateNonTiled (float *MatNext, float *MatPrev, unsigned int nCols, unsigned int NRows, unsigned int topRows, unsigned int botRows){

    int tc = threadIdx.x;
    int tr = threadIdx.y;
    int Row = blockIdx.y * blockDim.y + tr;
    int Col = blockIdx.x * blockDim.x + tc;

    if (Row < NRows && Col < nCols)
    {
        if ((Col>0 && Col <(nCols-1)) && (Row>=topRows && Row<=(NRows-botRows-1)))
        {
            float nord= MatPrev[(Row-1)*nCols+Col];
            float sud= MatPrev[(Row+1)*nCols+Col];
            float est= MatPrev[Row*nCols+Col+1];
            float ovest= MatPrev[Row*nCols+Col-1];

            float nw=MatPrev[(Row-1)*nCols+Col-1];
            float ne=MatPrev[(Row-1)*nCols+Col+1];
            float sw=MatPrev[(Row+1)*nCols+Col-1];
            float se=MatPrev[(Row+1)*nCols+Col+1];

            float primaParz= (4.0f*(nord+sud+ovest+est))+ nw+ne+sw+se;
            float nextValue= (float) primaParz/20.0f;

            MatNext[Row*nCols+Col]=nextValue;
        }
    }
}

__global__ void updateTiled(float *MatNext, float *MatPrev, unsigned int nCols, unsigned int NRows, unsigned int topRows, unsigned int botRows, const unsigned int tileX, const int tileY){
    
    extern __shared__ float tile[]; 

    int tc = threadIdx.x;
    int tr = threadIdx.y;
    int Row = blockIdx.y * blockDim.y + tr;
    int Col = blockIdx.x * blockDim.x + tc;
    
    int tid = tr * tileX + tc; 

    float nextValue = 0.0f;
    
    if (Row < NRows && Col < nCols)
    {
        tile[tid] = MatPrev[Row*nCols+Col];
        __syncthreads();

        if ((Col>0 && Col <(nCols-1)) && (Row>=topRows && Row<=(NRows-botRows-1)))
        {
            
            if(tr > 0 && tr < tileY - 1 && tc > 0 && tc < tileX - 1) {
                float nord  = tile[(tr-1) * tileX + tc];
                float sud   = tile[(tr+1) * tileX + tc];
                float est   = tile[tr * tileX + (tc+1)];
                float ovest = tile[tr * tileX + (tc-1)];

                float nw = tile[(tr-1) * tileX + (tc-1)];
                float ne = tile[(tr-1) * tileX + (tc+1)];
                float sw = tile[(tr+1) * tileX + (tc-1)];
                float se = tile[(tr+1) * tileX + (tc+1)];

                float primaParz = (4.0f*(nord+sud+ovest+est)) + nw+ne+sw+se;
                nextValue = (float) primaParz/20.0f;
                
                MatNext[Row*nCols+Col] = nextValue;
            } 
           
        }
    }
}