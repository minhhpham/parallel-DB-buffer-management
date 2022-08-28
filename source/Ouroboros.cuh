#include <iostream>

#include "Ouroboros/include/device/Ouroboros_impl.cuh"
#include "Ouroboros/include/device/MemoryInitialization.cuh"
#include "Ouroboros/include/InstanceDefinitions.cuh"
#include "Ouroboros/include/Utility.h"
#include "Paging.cuh"
#include "ErrorHandler.cuh"


/* ---------------------- declaration and usage ----------------------------------------------------------------- */
// extern __device__ void *pageAddress(int pageID);    // return pointer to start of a page given pageID
extern __host__ void initMemoryManagement(int nGB=TOTAL_SIZE_GB_DEFAULT, int pageSize=PAGE_SIZE_DEFAULT);    // initialize the memory management system
extern __device__ void *getPage();
extern __device__ void freePage(void *ptr);
// extern __host__ void resetMemoryManager();          // free all pages and reset meta data
extern __host__ void prefillBuffer(float freePercentage); // prefill the buffer so that it only has freePercentage % of free page. freePercentage is in [0,1]
__device__ int d_instance_pagesize;
int instance_pagesize;
__device__ int d_instance_nGB;
int instance_nGB;
MultiOuroVLPQ memoryManager;
__device__ Ouroboros<VLPQ, OuroborosPages<PageQueueVL, Chunk<8192UL>, 16384U, 2U>> *d_memoryManager;


void initMemoryManagement(int nGB, int pageSize){
    memoryManager.initialize(nGB * 1024ULL * 1024ULL * 1024ULL);
    auto d_memoryManager_h = memoryManager.getDeviceMemoryManager();
    gpuErrchk( cudaMemcpyToSymbol(d_memoryManager, &d_memoryManager_h, sizeof(void*)) );
    gpuErrchk( cudaMemcpyToSymbol(d_instance_pagesize, &pageSize, sizeof(int)) );
    gpuErrchk( cudaMemcpyToSymbol(d_instance_nGB, &nGB, sizeof(int)) );
    instance_pagesize = pageSize;
    instance_nGB = nGB;
}

__device__ void *getPage(){
    return reinterpret_cast<void*>(d_memoryManager->malloc(d_instance_pagesize));
}

__device__ void freePage(void* ptr){
    d_memoryManager->free(ptr);
}


__global__ void get1page_kernel(int Nthreads){
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if (tid<Nthreads)
		void *ptr = getPage();
}

__host__ void prefillBuffer(float freePercentage){
    unsigned long long h_total_n_pages = instance_nGB * 1024ULL * 1024ULL * 1024ULL / instance_pagesize;
    int nRequests = (1-freePercentage)*h_total_n_pages;
	if (nRequests==0) return;
	// std::cerr << "number of pages requested to fill buffer: " << nRequests << std::endl;
	// std::cerr << "filling buffer ...  " << std::endl;
	// std::cerr.flush();
	get1page_kernel <<< ceil((float)nRequests/128), 128 >>> (nRequests);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
	// std::cerr << "free page percentage: " << 100*getFreePagePercentage() << std::endl;
}