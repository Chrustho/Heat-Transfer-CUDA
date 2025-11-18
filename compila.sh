#!/bin/bash


nvcc -O3 -arch=sm_53 kernel.cu main.cu init.cu -Iinclude -o main
