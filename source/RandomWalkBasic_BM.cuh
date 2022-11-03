/*************** BASIC RANDOM WALK WITH BITMAP 32-bit ***************************/

#ifndef RWBASIC_BM_CUH
#define RWBASIC_BM_CUH

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
__device__ int *d_PageMapRandomWalk_BM;     // length TOTAL_N_PAGES/32
int *d_PageMapRandomWalk_BM_h;              // same pointer on host
static int Bitmap_length;
static __device__ int Bitmap_length_d;

/* RANDOM WALK IMPLEMENTATION */

/* 
    - initialize actual memory pages on GPU 
    - initialize memory management's metadata
        - bitmap (save same pointer on host and device variables)
        - length (same value on host and device)
    - set all memory free
 */
__host__ void initMemoryManagement(int nGB, int pageSize){
    // initialize actual pages
    initPages(nGB, pageSize);
    // initialize metadata (page map)
        // bitmap length
    Bitmap_length = h_total_n_pages/sizeof(int)/8;
    gpuErrchk( cudaMemcpyToSymbol(Bitmap_length_d, &Bitmap_length, sizeof(int)) );
        // bitmap
    gpuErrchk( cudaMalloc((void**)&d_PageMapRandomWalk_BM_h, Bitmap_length*sizeof(int)) );
    // copy to the global variable on device
    gpuErrchk( cudaMemcpyToSymbol(d_PageMapRandomWalk_BM, &d_PageMapRandomWalk_BM_h, sizeof(int*)) );
    // set all memory free
    gpuErrchk( cudaMemset(d_PageMapRandomWalk_BM_h, 0, Bitmap_length*sizeof(int)) );
}



// if step_count is not null, write step count to it
__device__ int getPage(int *stepCount){
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    int16_t Clock = (int16_t)clock();
    int seed = (tid<<15) + Clock;
    // randomize pages and try to grab a page
    int step_count = 0;
    int pageID=-1;
    while(pageID==-1){
        step_count++;
        // perform random jump
        seed = RNG_LCG(seed);
        unsigned groupID = (unsigned)seed % Bitmap_length_d;
        while (d_PageMapRandomWalk_BM[groupID]!=0xffffffff){
            int groupValue = d_PageMapRandomWalk_BM[groupID];
            // try to flip one of the 0-bit to 1
            int bitPosition = __ffs(~groupValue) - 1;
            int result = atomicOr(&d_PageMapRandomWalk_BM[groupID], 1<<bitPosition);
            // check if we actually flipped the bit
            if ((result & (1<<bitPosition)) == 0){
                // we flipped the bit
                // calculate the corresponding pageID
                pageID = groupID<<5 + bitPosition;
                break; // break the inner while loop
            }
        }
    }
    
    if (stepCount) *stepCount = step_count;
    return pageID;
}


__device__ void freePage(int pageID){
    int groupID = pageID/32;
    int bitPosition = pageID - (groupID*32);
    atomicAnd(&d_PageMapRandomWalk_BM[groupID], ~(1<<bitPosition) ); 
}

__host__ void resetMemoryManager(){
    gpuErrchk( cudaMemset(d_PageMapRandomWalk_BM_h, 0, Bitmap_length*sizeof(int)) );
}


__global__ void popc_kernel(int *d_output){
    /* kernel run with Bitmap_length threads */
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i>=Bitmap_length_d) return;
    d_output[i] = __popc(d_PageMapRandomWalk_BM[i]);

}

__host__ float getFreePagePercentage(){
    // first perform popc on all word on d_PageMapRandomWalk_BM
    int *d_popc_out;
    gpuErrchk( cudaMalloc(&d_popc_out, Bitmap_length*sizeof(int)) );
    popc_kernel <<< ceil((float)Bitmap_length/32), 32 >>> (d_popc_out);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    // use cub to find the sum of d_PageMapRandomWalkBasic
    int h_sum, *d_sum;    // output on host and on device
    gpuErrchk( cudaMalloc(&d_sum, sizeof(int)) );
    
    // first run to find memory requirement
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_popc_out, d_sum, Bitmap_length);
    // Allocate temporary storage
    gpuErrchk( cudaMalloc(&d_temp_storage, temp_storage_bytes) );
    // Run reduction
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_popc_out, d_sum, Bitmap_length);
    
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    // transfer output to host
    gpuErrchk( cudaMemcpy(&h_sum, d_sum, sizeof(int), cudaMemcpyDeviceToHost) );
    cudaFree(d_sum); cudaFree(d_temp_storage); cudaFree(d_popc_out);
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
	get1page_kernel <<< ceil((float)nRequests/32), 32 >>> (nRequests);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
	std::cerr << "free page percentage: " << 100*getFreePagePercentage() << std::endl;
}


#endif