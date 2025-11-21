#include "include/kernel.cuh"

__global__ void updateGlobal (float *MatNext, float *MatPrev, 
                                unsigned int nCols, unsigned int NRows, 
                                unsigned int topRows, unsigned int botRows){

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
            float nextValue= (float) primaParz*0.05f;

            MatNext[Row*nCols+Col]=nextValue;
        }
    }
}

__global__ void updateTiled(float *MatNext, float *MatPrev, 
                                            unsigned int nCols, unsigned int NRows, 
                                            unsigned int topRows, unsigned int botRows, 
                                            const unsigned int tileX, const int tileY) {
    
    extern __shared__ float tile[]; 

    int tc = threadIdx.x;
    int tr = threadIdx.y;
    int Row = blockIdx.y * blockDim.y + tr;
    int Col = blockIdx.x * blockDim.x + tc;
    
    int tid = tr * tileX + tc; 
    int globalIdx = Row * nCols + Col;

    if (Row < NRows && Col < nCols) {
        tile[tid] = MatPrev[globalIdx];
    }
    __syncthreads();

    if ((Col > 0 && Col < (nCols - 1)) && (Row >= topRows && Row <= (NRows - botRows - 1)))
    {
        float nord, sud, est, ovest, nw, ne, sw, se;

        if (tr > 0) nord = tile[(tr - 1) * tileX + tc];
        else        nord = MatPrev[(Row - 1) * nCols + Col];

        if (tr < tileY - 1) sud = tile[(tr + 1) * tileX + tc];
        else                sud = MatPrev[(Row + 1) * nCols + Col];

        if (tc > 0) ovest = tile[tr * tileX + (tc - 1)];
        else        ovest = MatPrev[Row * nCols + (Col - 1)];

        if (tc < tileX - 1) est = tile[tr * tileX + (tc + 1)];
        else                est = MatPrev[Row * nCols + (Col + 1)];


        if (tr > 0 && tc > 0) nw = tile[(tr - 1) * tileX + (tc - 1)];
        else                  nw = MatPrev[(Row - 1) * nCols + (Col - 1)];
        
        if (tr > 0 && tc < tileX - 1) ne = tile[(tr - 1) * tileX + (tc + 1)];
        else                          ne = MatPrev[(Row - 1) * nCols + (Col + 1)];

        if (tr < tileY - 1 && tc > 0) sw = tile[(tr + 1) * tileX + (tc - 1)];
        else                          sw = MatPrev[(Row + 1) * nCols + (Col - 1)];

        if (tr < tileY - 1 && tc < tileX - 1) se = tile[(tr + 1) * tileX + (tc + 1)];
        else                                  se = MatPrev[(Row + 1) * nCols + (Col + 1)];

        float primaParz = (4.0f * (nord + sud + ovest + est)) + nw + ne + sw + se;
        MatNext[globalIdx] = primaParz * 0.05f;
    }
}

__global__ void updateTiledPadding(float *MatNext, float *MatPrev, 
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

        n = (tr > 0)          ? tile[tid - paddedWidth] : MatPrev[idx_n_base + Col];
        s = (tr < tileY - 1)  ? tile[tid + paddedWidth] : MatPrev[idx_s_base + Col];
        w = (tc > 0)          ? tile[tid - 1]           : MatPrev[globalIdx - 1];
        e = (tc < tileX - 1)  ? tile[tid + 1]           : MatPrev[globalIdx + 1];

        nw = (tr > 0 && tc > 0)                 ? tile[tid - paddedWidth - 1] : MatPrev[idx_n_base + Col - 1];
        ne = (tr > 0 && tc < tileX - 1)         ? tile[tid - paddedWidth + 1] : MatPrev[idx_n_base + Col + 1];
        sw = (tr < tileY - 1 && tc > 0)         ? tile[tid + paddedWidth - 1] : MatPrev[idx_s_base + Col - 1];
        se = (tr < tileY - 1 && tc < tileX - 1) ? tile[tid + paddedWidth + 1] : MatPrev[idx_s_base + Col + 1];

        float primaParz = (4.0f * (n + s + w + e)) + nw + ne + sw + se;
        MatNext[globalIdx] = primaParz * 0.05f;
    }
}



__global__ void updateTiled_wH(float *MatNext, const float *MatPrev, 
                                   int nCols, int nRows, 
                                   int topRows, int botRows) {

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockDim.x;
    int by = blockDim.y;

    int col = blockIdx.x * bx + tx;
    int row = blockIdx.y * by + ty;
    
    extern __shared__ float s_tile[];
    int s_w = bx + 2;  

    int s_idx = (ty + 1) * s_w + (tx + 1);
    int globalIdx = row * nCols + col;

    float myVal = 0.0f;
    if (row < nRows && col < nCols) {
        myVal = MatPrev[globalIdx];
        s_tile[s_idx] = myVal;
    } else {
        s_tile[s_idx] = 0.0f;
    }


    if (ty == 0) {
        int r_prev = row - 1;
        if (r_prev >= 0) {
            if (col < nCols) 
                s_tile[tx + 1] = MatPrev[r_prev * nCols + col];
            else 
                s_tile[tx + 1] = 0.0f;

            if (tx == 0) {
                int c_prev = col - 1;
                s_tile[0] = (c_prev >= 0) ? MatPrev[r_prev * nCols + c_prev] : 0.0f;
            }
            if (tx == bx - 1) {
                int c_next = col + 1;
                s_tile[bx + 1] = (c_next < nCols) ? MatPrev[r_prev * nCols + c_next] : 0.0f;
            }
        } else {
            // Bordo superiore del dominio: tutto a 0
            s_tile[tx + 1] = 0.0f;
            if (tx == 0) s_tile[0] = 0.0f;
            if (tx == bx - 1) s_tile[bx + 1] = 0.0f;
        }
    }

    if (ty == by - 1) {
        int r_next = row + 1;
        int s_offset = (by + 1) * s_w; 
        
        if (r_next < nRows) {
            if (col < nCols)
                s_tile[s_offset + tx + 1] = MatPrev[r_next * nCols + col];
            else
                s_tile[s_offset + tx + 1] = 0.0f;

            if (tx == 0) {
                int c_prev = col - 1;
                s_tile[s_offset] = (c_prev >= 0) ? MatPrev[r_next * nCols + c_prev] : 0.0f;
            }
            if (tx == bx - 1) {
                int c_next = col + 1;
                s_tile[s_offset + bx + 1] = (c_next < nCols) ? MatPrev[r_next * nCols + c_next] : 0.0f;
            }
        } else {
            s_tile[s_offset + tx + 1] = 0.0f;
            if (tx == 0) s_tile[s_offset] = 0.0f;
            if (tx == bx - 1) s_tile[s_offset + bx + 1] = 0.0f;
        }
    }

    if (tx == 0) {
        int c_prev = col - 1;
        int s_pos = (ty + 1) * s_w; 
        if (c_prev >= 0 && row < nRows)
            s_tile[s_pos] = MatPrev[row * nCols + c_prev];
        else
            s_tile[s_pos] = 0.0f;
    }

    if (tx == bx - 1) {
        int c_next = col + 1;
        int s_pos = (ty + 1) * s_w + (bx + 1); 
        if (c_next < nCols && row < nRows)
            s_tile[s_pos] = MatPrev[row * nCols + c_next];
        else
            s_tile[s_pos] = 0.0f;
    }

    __syncthreads(); 


    if (row >= topRows && row < (nRows - botRows) && col > 0 && col < (nCols - 1)) {
        

        float* s_row_ptr = &s_tile[(ty + 1) * s_w]; 
        
        float n  = s_tile[ty * s_w + (tx + 1)];   
        float s  = s_tile[(ty + 2) * s_w + (tx + 1)]; 
        float w  = s_row_ptr[tx];                  
        float e  = s_row_ptr[tx + 2];               

        float nw = s_tile[ty * s_w + tx];          
        float ne = s_tile[ty * s_w + (tx + 2)];     
        float sw = s_tile[(ty + 2) * s_w + tx];     
        float se = s_tile[(ty + 2) * s_w + (tx + 2)]; 
        

        float res = (4.0f * (n + s + w + e) + nw + ne + sw + se) * 0.05f;
        
        MatNext[globalIdx] = res;
    }
}

