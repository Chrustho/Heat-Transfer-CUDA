#pragma once
__global__ void initTemperature(float *mat, 
                                unsigned int rows, 
                                unsigned int cols, 
                                unsigned int initTemp, 
                                unsigned int topRows, 
                                unsigned int botRows);