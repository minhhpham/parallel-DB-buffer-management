/*************** COLLABORATIVE RANDOM WALK WITH BITMAP 32-bit ***************************/

#ifndef COLLABRW_CUH
#define COLLABRW_CUH

#include <stdio.h>
#include "ErrorHandler.cuh"
#include "Paging.cuh"
#include "RNG_LCG.cuh"
#include <cub/cub.cuh>


/* ---------------------- declaration and usage ----------------------------------------------------------------- */
extern __device__ void *pageAddress(int pageID);    // return pointer to start of a page given pageID
__host__ void initMemoryManagement(int nGB=TOTAL_SIZE_GB_DEFAULT, int pageSize=PAGE_SIZE_DEFAULT);    // initialize the memory management system
extern __device__ int getPage(int *step_count=0, bool collectWarpData=false);
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


/* release consecutive pages */
// given groupID, bitposition, nPages
static inline __device__ void releasePagesBase(int groupID, int bitPosition, int nPages){
	int flipMask = ~(((1<<nPages)-1)<<bitPosition);
	atomicAnd(&d_PageMapRandomWalk_BM[groupID], flipMask);
}

// given pageID, nPages. They all should belong to the same bit group
static inline __device__ void releasePages(int pageID, int nPages){
	int groupID = pageID/32;
	int bitPosition = pageID - groupID*32;
	releasePagesBase(groupID, bitPosition, nPages);
}

__device__ void freePage(int pageID){
    int groupID = pageID/32;
    int bitPosition = pageID - (groupID*32);
    atomicAnd(&d_PageMapRandomWalk_BM[groupID], ~(1<<bitPosition) ); 
}



static inline __device__ int fks(int x, int k){
	// find kth bit set on x, first bit is 1, last bit is 32
	// return 0 if x==0 or there are less than k set bit
    if (__popc(x)<k) return 0;
	int left, right;
	left = 1, right = 32;
	while (left<right){
		int mid = (left+right)/2;
		int count = __popc( ((1<<mid)-1) & x ); // count number of bits from bit 1 to bit mid
		if (count<k) left = mid + 1;
		else right = mid;
	}
	return left;
}

/*
	each thread found f pages:
	warp reduce to find maxF
	for i = 0 to maxF:
		if current thread has a page to give:
			hasPage = True
		if current thread sill needs a page:
			needPage = True
		hasPageMask = ballot_sync(activemask, hasPage)
		needPageMask = ballot_sync(activemask, needPage)
		b = number of set bits in needPageMask before position of (this laneID)
			( = popc( ~(0xFFFFFFFF<<laneID) & needPageMask ) )
		sourceLaneID = fbs(hasPageMask, b+1)
			find (b+1)th set (ffs = find first set)
			position of the bth set bit on needPageMask
			( need to do O(32) ffs )
		shuffle sync to get pageID from sourceLaneID
*/
__device__ int getPage(int *stepCount, bool collectWarpData){
	int16_t Clock = (int16_t)clock();
	int seed = (blockIdx.x<<15) + Clock;
	// randomize pages and try to grab a page
	int step_count = 0;
	unsigned mask = __activemask();
	int needMask = mask;
	int laneID = threadIdx.x%32;
	int pageID_out = -1;
	while (needMask){
		step_count++;
		seed = RNG_LCG(seed);
		unsigned p = (unsigned)(seed + threadIdx.x) % Bitmap_length_d; // groupID
		int r = atomicExch(&d_PageMapRandomWalk_BM[p], 0xFFFFFFFF);
		r = ~r;
		int hasMask = __ballot_sync(mask, r!=0);

		if (collectWarpData){
			int nPagesFound = __popc(r);
			int Sum = 0;
			for (int tid = 0; tid<32; tid++)
				Sum += __shfl_sync(mask, nPagesFound, tid);
			if (threadIdx.x==0)
				printf("%d\n", Sum);
		}
		
		while (needMask && hasMask){
			int s = __popc( ~(0xFFFFFFFF<<laneID) & needMask );
			int sourceLaneID = fks(hasMask, s+1)-1;
			int foundPageID = p*32 + __ffs(r)-1;
			int pageID = __shfl_sync(mask, foundPageID, sourceLaneID);
			if (pageID_out==-1 && sourceLaneID!=-1)
				pageID_out = pageID;
			// update pages this thread has, if a page has been taken from it
			if ( r!=0 && __popc( ~(0xFFFFFFFF<<laneID) & hasMask ) <  __popc(needMask) )
				r &= ~(1<<(__ffs(r)-1)) ;
			// update needmask and hasmask
			hasMask = __ballot_sync(mask, r!=0);
			needMask = __ballot_sync(mask, pageID_out==-1);
		}
		// release page we're still holding
		if (needMask==0 && r)
			atomicExch(&d_PageMapRandomWalk_BM[p], ~r);
	}
	if (stepCount) *stepCount = step_count;
	return pageID_out;
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
	auto free_perc = getFreePagePercentage();
	std::cerr << "free page percentage: " << (int)(free_perc * h_total_n_pages) << "/" << h_total_n_pages << " = " << 100*getFreePagePercentage() << std::endl;
}


#endif