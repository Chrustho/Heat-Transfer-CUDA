#pragma once
__global__ void updateNonTiled (float *MatNext, 
                                float *MatPrev, 
                                unsigned int nCols, 
                                unsigned int NRows, 
                                unsigned int topRows, 
                                unsigned int botRows);
