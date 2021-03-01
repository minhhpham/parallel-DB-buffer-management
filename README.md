# Buffer Management in Massively Parallel Systems

## Description
This is code for implementation of scalable buffer managers in GPUs. Unit Tests are also included in files `unitTest*`.

## How to use
 - Copy `parallelPage.cu` and `parallelPage.cuh` to your project
 - `#include "parallelpage.cuh"` in your code
 - You need to do separate compiling and linking as follows:
 ```
 nvcc --device-c -arch=sm_70 -rdc=true -lcudadevrt parallelPage.cu -o parallelPage.o
 nvcc --device-c -arch=sm_70 $(your_code).cu -o $(your_code).o
 nvcc -arch=sm_70 parallelPage.o $(your_code).o -o $(binary_file)
 ```

## Citation
TBA