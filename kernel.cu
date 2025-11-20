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
            float nextValue= (float) primaParz*0.05f;

            MatNext[Row*nCols+Col]=nextValue;
        }
    }
}

__global__ void updateTiledOptimizedNormale(float *MatNext, float *MatPrev, 
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

        bool isInternal = (tr > 0) && (tr < tileY - 1) && (tc > 0) && (tc < tileX - 1);

        if (isInternal) 
        {

            nord  = tile[(tr - 1) * tileX + tc];
            sud   = tile[(tr + 1) * tileX + tc];
            ovest = tile[tr * tileX + (tc - 1)];
            est   = tile[tr * tileX + (tc + 1)];

            nw    = tile[(tr - 1) * tileX + (tc - 1)];
            ne    = tile[(tr - 1) * tileX + (tc + 1)];
            sw    = tile[(tr + 1) * tileX + (tc - 1)];
            se    = tile[(tr + 1) * tileX + (tc + 1)];
        } 
        else 
        {

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
        }

        float primaParz = (4.0f * (nord + sud + ovest + est)) + nw + ne + sw + se;
        MatNext[globalIdx] = primaParz * 0.05f;
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
        MatNext[globalIdx] = primaParz * 0.05f;
    }
}

__global__ void tiled_wH(float *MatNext, float *MatPrev, 
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

    int i_halo= Row-1;
    int j_halo= Col-1;

    if ((0<=i_halo)&&(i_halo<NRows) && (0<=j_halo) &&(j_halo<nCols))
    {
        tile[tid]=MatPrev[i_halo*nCols+j_halo];
    }else{
        tile[tid]=0.0f;
    }

    __syncthreads();

    if ((Col > 0 && Col < (nCols - 1)) && (Row >= topRows && Row <= (NRows - botRows - 1)))
    {
        float nord, sud, est, ovest, nw, ne, sw, se;

        nord  = tile[(tr - 1) * tileX + tc];
        sud   = tile[(tr + 1) * tileX + tc];
        ovest = tile[tr * tileX + (tc - 1)];
        est   = tile[tr * tileX + (tc + 1)];

        nw    = tile[(tr - 1) * tileX + (tc - 1)];
        ne    = tile[(tr - 1) * tileX + (tc + 1)];
        sw    = tile[(tr + 1) * tileX + (tc - 1)];
        se    = tile[(tr + 1) * tileX + (tc + 1)];
        
        float primaParz = (4.0f * (nord + sud + ovest + est)) + nw + ne + sw + se;
        MatNext[globalIdx] = primaParz * 0.05f;
    }

}

__global__ void tiled_wH_corrected(float *MatNext, const float *MatPrev, 
                                   int nCols, int nRows, 
                                   int topRows, int botRows) {

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockDim.x;
    int by = blockDim.y;

    int col = blockIdx.x * bx + tx;
    int row = blockIdx.y * by + ty;
    
    int globalIdx = row * nCols + col;

    extern __shared__ float s_tile[];

    int s_w = bx + 2;

    int s_idx_center = (ty + 1) * s_w + (tx + 1);

    if (row < nRows && col < nCols) {
        s_tile[s_idx_center] = MatPrev[globalIdx];
    } else {
        s_tile[s_idx_center] = 0.0f; 
    }

    
    if (ty == 0) {
        int r_in = row - 1;
        if (r_in >= 0 && col < nCols) 
            s_tile[0 * s_w + (tx + 1)] = MatPrev[r_in * nCols + col];
        else 
            s_tile[0 * s_w + (tx + 1)] = 0.0f;
    }
    if (ty == by - 1) {
        int r_in = row + 1;
        if (r_in < nRows && col < nCols) 
            s_tile[(by + 1) * s_w + (tx + 1)] = MatPrev[r_in * nCols + col];
        else 
            s_tile[(by + 1) * s_w + (tx + 1)] = 0.0f;
    }
    if (tx == 0) {
        int c_in = col - 1;
        if (c_in >= 0 && row < nRows)
            s_tile[(ty + 1) * s_w + 0] = MatPrev[row * nCols + c_in];
        else
            s_tile[(ty + 1) * s_w + 0] = 0.0f;
    }
    if (tx == bx - 1) {
        int c_in = col + 1;
        if (c_in < nCols && row < nRows)
            s_tile[(ty + 1) * s_w + (bx + 1)] = MatPrev[row * nCols + c_in];
        else
            s_tile[(ty + 1) * s_w + (bx + 1)] = 0.0f;
    }

    if (tx == 0 && ty == 0) {
        int r_in = row - 1; int c_in = col - 1;
        if (r_in >= 0 && c_in >= 0) s_tile[0] = MatPrev[r_in*nCols + c_in];
        else s_tile[0] = 0.0f;
    }

    if (tx == bx - 1 && ty == 0) {
        int r_in = row - 1; int c_in = col + 1;
        if (r_in >= 0 && c_in < nCols) s_tile[bx + 1] = MatPrev[r_in*nCols + c_in];
        else s_tile[bx + 1] = 0.0f;
    }
    if (tx == 0 && ty == by - 1) {
        int r_in = row + 1; int c_in = col - 1;
        if (r_in < nRows && c_in >= 0) s_tile[(by+1)*s_w] = MatPrev[r_in*nCols + c_in];
        else s_tile[(by+1)*s_w] = 0.0f;
    }
    if (tx == bx - 1 && ty == by - 1) {
        int r_in = row + 1; int c_in = col + 1;
        if (r_in < nRows && c_in < nCols) s_tile[(by+1)*s_w + (bx+1)] = MatPrev[r_in*nCols + c_in];
        else s_tile[(by+1)*s_w + (bx+1)] = 0.0f;
    }

    __syncthreads(); 

    if (row >= topRows && row < (nRows - botRows) && col > 0 && col < (nCols - 1)) {
        
        int c = (ty + 1) * s_w + (tx + 1);
        
        float nord  = s_tile[c - s_w];
        float sud   = s_tile[c + s_w];
        float ovest = s_tile[c - 1];
        float est   = s_tile[c + 1];

        float nw    = s_tile[c - s_w - 1];
        float ne    = s_tile[c - s_w + 1];
        float sw    = s_tile[c + s_w - 1];
        float se    = s_tile[c + s_w + 1];
        
        float res = (4.0f * (nord + sud + ovest + est) + nw + ne + sw + se) * 0.05f;
        MatNext[globalIdx] = res;
    }
}