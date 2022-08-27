/******** BASIC RANDOM WALK IMPLEMENTATION **************/


#ifndef RWBASIC_CUH
#define RWBASIC_CUH

#include <stdio.h>
#include "ErrorHandler.cuh"
#include "Paging.cuh"
#include "RNG_LCG.cuh"
#include <cub/cub.cuh>


/* ---------------------- declaration and usage ----------------------------------------------------------------- */
extern __device__ void *pageAddress(int pageID);    // return pointer to start of a page given pageID
extern __host__ void initMemoryManagement(int nGB=TOTAL_SIZE_GB_DEFAULT, int pageSize=PAGE_SIZE_DEFAULT);    // initialize the memory management system
extern __device__ int getPage(int *step_count=0);
extern __device__ void freePage(int pageID);
extern __host__ float getFreePagePercentage();
extern __host__ void resetMemoryManager();          // free all pages and reset meta data
extern __host__ void prefillBuffer(float freePercentage); // prefill the buffer so that it only has freePercentage % of free page. freePercentage is in [0,1]



/* -------------------------- definitions -------------------------------------------------- */
/* data structures for Random Walk*/
// for each element: 0 means page is free, 1 means page is occupied
static __device__ int *d_PageMapRandomWalkBasic;
static int *d_PageMapRandomWalkBasic_h;   // same pointer on host access

/* RANDOM WALK IMPLEMENTATION */

/* 
    - initialize actual memory pages on GPU 
    - initialize memory management's metadata
    - set all memory free
 */
__host__ void initMemoryManagement(int nGB, int pageSize){
    // initialize actual pages
    initPages(nGB, pageSize);
    // initialize metadata (page map)
    gpuErrchk( cudaMalloc((void**)&d_PageMapRandomWalkBasic_h, h_total_n_pages*sizeof(int)) );
    // copy to the global variable on device
    gpuErrchk( cudaMemcpyToSymbol(d_PageMapRandomWalkBasic, &d_PageMapRandomWalkBasic_h, sizeof(int*)) );
    // set all memory free
    gpuErrchk( cudaMemset(d_PageMapRandomWalkBasic_h, 0, h_total_n_pages*sizeof(int)) );
}



// if step_count is not null, write step count to it
__device__ int d_counter = 0;
__device__ int getPage(int *stepCount){
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    int16_t Clock = (int16_t)clock();
    int seed = (tid<<15) + Clock;
    seed = RNG_LCG(seed);
    // randomize pages and try
    unsigned pageID = (unsigned)seed % d_total_n_pages;
    int step_count = 1;
    while(atomicExch(&d_PageMapRandomWalkBasic[pageID],1) == 1){
        seed = RNG_LCG(seed);
        pageID = (unsigned)seed % d_total_n_pages;
        step_count++;
        if (step_count>d_total_n_pages/2){
            printf("BUFFER MANAGER OUT OF PAGE !!!\n");
            __trap();
        }
    }

    if (stepCount) *stepCount = step_count;
    return pageID;
}


__device__ void freePage(int pageID){
    atomicExch(&d_PageMapRandomWalkBasic[pageID], 0);
}

__host__ void resetMemoryManager(){
    gpuErrchk( cudaMemset(d_PageMapRandomWalkBasic_h, 0, h_total_n_pages*sizeof(int)) );
}



__host__ float getFreePagePercentage(){
    // use cub to find the sum of d_PageMapRandomWalkBasic
    int h_sum, *d_sum;    // output on host and on device
    gpuErrchk( cudaMalloc(&d_sum, sizeof(int)) );
    
    // first run to find memory requirement
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_PageMapRandomWalkBasic_h, d_sum, h_total_n_pages);
    // Allocate temporary storage
    gpuErrchk( cudaMalloc(&d_temp_storage, temp_storage_bytes) );
    // Run reduction
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_PageMapRandomWalkBasic_h, d_sum, h_total_n_pages);
    
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    // transfer output to host
    gpuErrchk( cudaMemcpy(&h_sum, d_sum, sizeof(int), cudaMemcpyDeviceToHost) );
    cudaFree(d_sum); cudaFree(d_temp_storage);
    return (float)(h_total_n_pages-h_sum)/h_total_n_pages;
}

__global__ void get1page_kernel(int Nthreads){
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if (tid<Nthreads)
		int pageID = getPage(nullptr);
}

__host__ void prefillBuffer(float freePercentage){
    int nRequests = (1-freePercentage)*h_total_n_pages;
	if (nRequests==0) return;
	// std::cerr << "number of pages requested to fill buffer: " << nRequests << std::endl;
	// std::cerr << "filling buffer ...  " << std::endl;
	// std::cerr.flush();
	get1page_kernel <<< ceil((float)nRequests/1024), 1024 >>> (nRequests);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
	// std::cerr << "free page percentage: " << 100*getFreePagePercentage() << std::endl;
}


#endif