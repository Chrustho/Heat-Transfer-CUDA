#pragma once

__global__ void updateGlobal (float *MatNext, 
                                float *MatPrev, 
                                unsigned int nCols, 
                                unsigned int NRows, 
                                unsigned int topRows, 
                                unsigned int botRows);

__global__ void updateTiled(float *MatNext, float *MatPrev, 
                                     unsigned int nCols, unsigned int NRows, 
                                     unsigned int topRows, unsigned int botRows, 
                                     const unsigned int tileX, const int tileY);
__global__ void updateTiledPadding(float *MatNext, float *MatPrev, 
                                     unsigned int nCols, unsigned int NRows, 
                                     unsigned int topRows, unsigned int botRows, 
                                     const unsigned int tileX, const int tileY);

__global__ void updateTiled_wH(float *MatNext, const float *MatPrev, 
                                   int nCols, int nRows, 
                                   int topRows, int botRows);



