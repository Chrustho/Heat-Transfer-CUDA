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

__global__ void updateTiledOptimized(float *MatNext, float *MatPrev, 
                                     unsigned int nCols, unsigned int NRows, 
                                     unsigned int topRows, unsigned int botRows, 
                                     const unsigned int tileX, const int tileY) {
    

    extern __shared__ float tile[]; 

    int tc = threadIdx.x;
    int tr = threadIdx.y;
    int Row = blockIdx.y * blockDim.y + tr;
    int Col = blockIdx.x * blockDim.x + tc;
    
    int paddedWidth = tileX + 1;
    
    int tid = tr * paddedWidth + tc; 
    
    int globalIdx = Row * nCols + Col;

    if (Row < NRows && Col < nCols) {
        tile[tid] = MatPrev[globalIdx];
    }
    __syncthreads();

    if ((Col > 0 && Col < (nCols - 1)) && (Row >= topRows && Row <= (NRows - botRows - 1)))
    {
        float n, s, e, w, nw, ne, sw, se;

        int idx_n_base = (Row - 1) * nCols;
        int idx_s_base = (Row + 1) * nCols;

        if (tr > 0 && tr < tileY - 1 && tc > 0 && tc < tileX - 1) 
        {
            n  = tile[tid - paddedWidth];
            s  = tile[tid + paddedWidth];
            w  = tile[tid - 1];
            e  = tile[tid + 1];

            nw = tile[tid - paddedWidth - 1];
            ne = tile[tid - paddedWidth + 1];
            sw = tile[tid + paddedWidth - 1];
            se = tile[tid + paddedWidth + 1];
        } 
        else 
        {

            n = (tr > 0)          ? tile[tid - paddedWidth] : MatPrev[idx_n_base + Col];
            s = (tr < tileY - 1)  ? tile[tid + paddedWidth] : MatPrev[idx_s_base + Col];
            w = (tc > 0)          ? tile[tid - 1]           : MatPrev[globalIdx - 1];
            e = (tc < tileX - 1)  ? tile[tid + 1]           : MatPrev[globalIdx + 1];

            nw = (tr > 0 && tc > 0)                 ? tile[tid - paddedWidth - 1] : MatPrev[idx_n_base + Col - 1];
            ne = (tr > 0 && tc < tileX - 1)         ? tile[tid - paddedWidth + 1] : MatPrev[idx_n_base + Col + 1];
            sw = (tr < tileY - 1 && tc > 0)         ? tile[tid + paddedWidth - 1] : MatPrev[idx_s_base + Col - 1];
            se = (tr < tileY - 1 && tc < tileX - 1) ? tile[tid + paddedWidth + 1] : MatPrev[idx_s_base + Col + 1];
        }

        float primaParz = (4.0f * (n + s + w + e)) + nw + ne + sw + se;
        MatNext[globalIdx] = primaParz / 20.0f;
    }
}