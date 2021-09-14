#include "parallelPage.cuh"
#include <stdint.h>
#include <cooperative_groups.h>
#include <time.h>
#include <stdlib.h>
#include <assert.h>

namespace cg = cooperative_groups;

#define GROUP_N_PAGES 100000   // group pages for faster initialization

/* actual pages for all strategies */
__device__ void **d_page_groups;                                       // actual address to pages, (TOTAL_N_PAGES/GROUP_N_PAGES) groups, each with GROUP_N_PAGES pages
__device__ void *pageAddress(int pageID){
	// find group id
	int groupID = pageID/GROUP_N_PAGES;
	// find offset on group
	int offset = pageID - groupID*GROUP_N_PAGES;
	// calculate actual address
	void *groupAddr = d_page_groups[groupID];
	return (void*)((char*)groupAddr + offset*PAGE_SIZE);
}


/* data structures for Random Walk*/
// for each element: 0 means page is free, 1 means page is occupied
static __device__ int *d_PageMapRandomWalk;
static __device__ int *d_RNG;
static __device__ int *d_RNG_idx;

/* data structures for Clustered Random Walk */
static __device__ int *d_LastFreePage;	// last free pageID obtained by each thread by calling freePageClusteredRandomWalk(), -1 if not initialized

/* data structures for Linked List */
typedef struct Node_t
{
	int pageID;
	Node_t *nextNode;	// null if none
} Node_t;
static __device__ Node_t *d_nodes;		// pre-allocated nodes
volatile static __device__ int d_HeadNodeID;		// index of headnode on d_nodes, -1 if it's not yet returned by a thread, -2 if no free page
volatile static __device__ int d_TailNodeID;		// index of tailnode on d_nodes, -1 if it's not yet returned by a thread

static __device__ long long unsigned d_LLticket;
static __device__ long long unsigned d_LLturn;
// static __device__ Node_t *d_HeadNode;	// pointer to first free node
// static __device__ int d_lockHeadNode;	// 0 means free
// static __device__ Node_t *d_TailNode;	// pointer to last free node
// static __device__ int d_lockTailNode;	// 0 means free

/* actual page initialization on GPU */
__host__ void initPages(){
	int n_groups = ceil((float)TOTAL_N_PAGES/GROUP_N_PAGES);
	printf("initializing %d groups on GPU, each with %d pages. Total = %ld MB \n", n_groups, GROUP_N_PAGES, (long)n_groups*GROUP_N_PAGES*PAGE_SIZE/1048576);
	// initialize (TOTAL_N_PAGES/GROUP_N_PAGES) groups on GPU memory, save their pointers on a host array
	void **h_groups = (void**)malloc(n_groups*sizeof(void*));
	for (int i=0; i<n_groups; i++){ // allocate each group
		gpuErrchk( cudaMalloc((void**)&h_groups[i], GROUP_N_PAGES*PAGE_SIZE) );
	}
	// allocate an array for d_pages and transfer the array of group pointers to this array
	void* tmp;
	cudaMalloc(&tmp, n_groups*sizeof(void*));
	gpuErrchk( cudaMemcpy(tmp, h_groups, n_groups*sizeof(void*), cudaMemcpyHostToDevice) );

	// set d_pages to this array
	gpuErrchk( cudaMemcpyToSymbol(d_page_groups, &tmp, sizeof(void*)) );

	// end
	free(h_groups);
}


/* RANDOM WALK IMPLEMENTATION */

/* Arrange the N elements of ARRAY in random order.
   Only effective if N is much smaller than RAND_MAX;
   if this may not be the case, use a better random
   number generator. */
static inline void shuffle(int *array, size_t n)
{
    if (n > 1) 
    {
        size_t i;
        for (i = 0; i < n - 1; i++) 
        {
          size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
          int t = array[j];
          array[j] = array[i];
          array[i] = t;
        }
    }
}

/* function to replace curand, better efficiency */
#define LCG_M 1<<31
#define LCG_A 1103515245
#define LCG_C 12345
__device__ static inline int RNG_LCG(int seed){
    long long seed_ = (long long)seed;
    return (int)((LCG_A*seed_ + LCG_C)%(LCG_M));
    // uint32_t x = seed;
    // x ^= x << 13;
    // x ^= x >> 17;
    // x ^= x << 5;
    // return x;
}

/* initialize TOTAL_N_PAGES pages on GPU's global memory, each page is PAGE_SIZE large
also initialize the page map structure with all 0 (all free) */
__host__ void initPagesRandomWalk(){
	// initialize actual pages
	initPages();

	// initialize page map
	void *h_PageMapRandomWalk;
	gpuErrchk( cudaMalloc((void**)&h_PageMapRandomWalk, TOTAL_N_PAGES*sizeof(int)) );
	gpuErrchk( cudaMemcpyToSymbol(d_PageMapRandomWalk, &h_PageMapRandomWalk, sizeof(void*)) );
	gpuErrchk( cudaMemset(h_PageMapRandomWalk, 0, TOTAL_N_PAGES*sizeof(int)) );

	// initialize random numbers
	int *h_RNG;
	gpuErrchk( cudaMalloc((void**)&h_RNG, TOTAL_N_PAGES*sizeof(int)) ); // allocate memory on device then copy to symbol
	gpuErrchk( cudaMemcpyToSymbol(d_RNG, &h_RNG, sizeof(void*)) );
	srand(time(NULL));
	int *tmp = (int*)malloc(TOTAL_N_PAGES*sizeof(int));
	for (int i=0; i<TOTAL_N_PAGES; i++)
		tmp[i] = i;
	shuffle(tmp, TOTAL_N_PAGES);
	// copy rng numbers to device
	gpuErrchk( cudaMemcpy(h_RNG, tmp, TOTAL_N_PAGES*sizeof(int), cudaMemcpyHostToDevice) );

	// initialize RNG index
	for (int i=0; i<TOTAL_N_PAGES; i++)
		tmp[i] = 0;
	// copy RNG index to device
	int *h_RNG_idx;
	gpuErrchk( cudaMalloc((void**)&h_RNG_idx, TOTAL_N_PAGES*sizeof(int)) ); // allocate memory on device then copy to symbol
	gpuErrchk( cudaMemcpyToSymbol(d_RNG_idx, &h_RNG_idx, sizeof(void*)) );
	gpuErrchk( cudaMemcpy(h_RNG_idx, tmp, TOTAL_N_PAGES*sizeof(int), cudaMemcpyHostToDevice) );

	free(tmp);
}



// if step_count is not null, write step count to it
__device__ int d_counter = 0;
__device__ int getPageRandomWalk(int *stepCount){
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	int16_t Clock = (int16_t)clock();
	int seed = (tid<<15) + Clock;
	seed = RNG_LCG(seed);
	// randomize pages and try
	int pageID = seed % TOTAL_N_PAGES;
	if (pageID<0) pageID = -pageID;
	int step_count = 1;
	while(atomicExch(&d_PageMapRandomWalk[pageID],1) == 1){
		seed = RNG_LCG(seed);
		pageID = seed % TOTAL_N_PAGES;
		step_count++;
	}

	// int pageID = tid*100;
	// atomicExch(&d_PageMapRandomWalk[pageID],1);
	// int step_count = 1;

	if (stepCount) *stepCount = step_count;
	return pageID;
}


__device__ void freePageRandomWalk(int pageID){
	atomicExch(&d_PageMapRandomWalk[pageID], 0);
}


__global__ void resetBufferRandomWalk_kernel(){
	memset(d_PageMapRandomWalk, 0, TOTAL_N_PAGES*sizeof(int));
}

__host__ void resetBufferRandomWalk(){
	resetBufferRandomWalk_kernel <<< 1, 1 >>> ();
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
}

__global__ void printNumPagesLeftRandomWalk_kernel(){
	int count = 0;
	for (int i=0; i<TOTAL_N_PAGES; i++){
		if (d_PageMapRandomWalk[i]==0) count++;
	}
	printf("[RW info] Number of free pages: %d \n", count);
}

__host__ void printNumPagesLeftRandomWalk(){
	printNumPagesLeftRandomWalk_kernel <<< 1, 1 >>> ();
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
}

/* CLUSTERED RANDOM WALK IMPLEMENTATION */
/*
 - initialize TOTAL_N_PAGES pages on GPU's global memory, each page is PAGE_SIZE large
 - initialize the page map structure with all 0 (all free) 
 - initialize the d_LastFreePage array to -1
 */
__host__ void initPagesClusteredRandomWalk(){
	initPagesRandomWalk();

	// init the last free page array
	void *d_tmp;	// global array, set to all -1
	gpuErrchk( cudaMalloc((void**)&d_tmp, 65536*1024*sizeof(int)) );	// allocate up to a grid's max number of threads
	gpuErrchk( cudaMemset(d_tmp, -1, TOTAL_N_PAGES*sizeof(int)) );
	// copy ptr value to symbol d_LastFreePage
	gpuErrchk( cudaMemcpyToSymbol(d_LastFreePage, &d_tmp, sizeof(void*)) );
}

// if step_count is not null, write step count to it
__device__ int getPageClusteredRandomWalk(int *stepCount){
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	// check page next to the last page visited
	int pageID = d_LastFreePage[tid];
	if (pageID!=-1){
		pageID++;
		if (pageID==TOTAL_N_PAGES) pageID = 0;
		if (d_PageMapRandomWalk[pageID]==0){
			if (atomicExch(&d_PageMapRandomWalk[pageID],1) == 0){	// success
				if (stepCount) *stepCount = 1;
				d_LastFreePage[tid] = pageID;
				return pageID;
			}
		}
	}
	// regular random walk
	pageID = getPageRandomWalk(stepCount);
	d_LastFreePage[tid] = pageID;
	return pageID;
}

// same free page function
__device__ void freePageClusteredRandomWalk(int pageID){
	atomicExch(&d_PageMapRandomWalk[pageID], 0);
}

__global__ void printNumPagesLeftClusteredRandomWalk_kernel(){
	// count free pages
	int count = 0;
	for (int i=0; i<TOTAL_N_PAGES; i++){
		if (d_PageMapRandomWalk[i]==0) count++;
	}
	printf("[CRW info] Number of free pages: %d \n", count);
	// count clusters
	int state = d_PageMapRandomWalk[0];
	if (state==0) count = 1;
	else count = 0;
	for (int i=1; i<TOTAL_N_PAGES; i++){
		if (state==0 && d_PageMapRandomWalk[i]==1)	// end of cluster
			state = 1;
		if (state==1 && d_PageMapRandomWalk[i]==0){	// start of cluster
			state = 0;
			count++;
		}
	}
	// check last page and first page if they are in one cluster
	if (state==0 && d_PageMapRandomWalk[0]==0) count--;
	printf("[CRW info] Number of clusters: %d \n", count);
}

__host__ void printNumPagesLeftClusteredRandomWalk(){
	printNumPagesLeftClusteredRandomWalk_kernel <<< 1, 1 >>> ();
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
}


/* LINKED LIST IMPLEMENTATION */
/* initialize TOTAL_N_PAGES pages on GPU's global memory, each page is PAGE_SIZE large
	also initialize the linked list structure, all pages free */
__global__ void initPagesLinkedList_kernel(){
	int i = blockIdx.x*blockDim.x + threadIdx.x; // nodeID
	if (i>=TOTAL_N_PAGES) return;
	d_nodes[i].pageID = i;	// pageID = nodeID
	if (i<TOTAL_N_PAGES-1)
		d_nodes[i].nextNode = &d_nodes[i+1];
	else
		d_nodes[i].nextNode = NULL; // last node has no next
	// set head node
	if (i==0)
		d_HeadNodeID = 0;	// first node is head
	// set tail node
	if (i==TOTAL_N_PAGES-1)	// last node is tail
		d_TailNodeID = TOTAL_N_PAGES-1;

	// init lock
	if (i==0){
		d_LLticket = 0;
		d_LLturn = 0;
	}
}

__host__ void initPagesLinkedList(){
	// initialize pages
	initPages();

	// initialize linked list 
	Node_t *h_d_nodes;	// allocate nodes array and get pointer value on host, then copy this value to d_nodes
	gpuErrchk( cudaMalloc((void**)&h_d_nodes, TOTAL_N_PAGES*sizeof(Node_t)) );
	gpuErrchk( cudaMemcpyToSymbol(d_nodes, &h_d_nodes, sizeof(Node_t*)) );

	initPagesLinkedList_kernel <<< ceil((float)TOTAL_N_PAGES/32), 32 >>> ();
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
}

// if step_count is not null, write step count to it
// atomically get d_HeadNodeID and replace with -1 until one thread replace it with a legit nodeID
// return -2 to d_HeadNodeID if no page is free
__device__ int getPageLinkedList(int *stepCount){
	unsigned long long start = (unsigned long long)clock();
	int nodeID = -1;
	int pageID;
	int step_count = 0;

	while (nodeID==-1){
		nodeID = atomicExch((int*)&d_HeadNodeID, -1);
		if (nodeID != -1){	// this thread successfully capture the head node
			// check if OOM
			if (nodeID==-2){printf("Out of Free Page!\n"); __trap();}
			// find pageID, should be equal to nodeID
			pageID = d_nodes[nodeID].pageID;
			if (pageID!=nodeID){printf("Fatal Error in getPageLinkedList: Inconsistent Node Data\n"); __trap();}
			int next_nodeID;
			if (d_nodes[nodeID].nextNode==NULL) next_nodeID = -2;	// no free page
			else next_nodeID = d_nodes[nodeID].nextNode->pageID;
			// invalidate obtained node's nextNode
			d_nodes[nodeID].nextNode = NULL;
			// return the next legit nodeID back to d_HeadNodeID
			atomicExch((int*)&d_HeadNodeID, next_nodeID);
		}
		
		step_count++;
	}
	unsigned long long stop = (unsigned long long)clock();
	if (stepCount) *stepCount = (int)(stop - start);
	return pageID;

	// FA MUTEX
	// volatile unsigned long long start = (unsigned long long)clock();
	// int pageID;
	// cg::grid_group grid = cg::this_grid();

	// volatile unsigned long long thread_ticket;
	// thread_ticket = atomicAdd(&d_LLticket, (unsigned long long)1);
	// grid.sync();

	// // access linked list
	// do {
	// 	if (thread_ticket == d_LLturn){
	// 		// find pageID, should be equal to nodeID
	// 		pageID = d_HeadNodeID;
	// 		d_HeadNodeID = d_nodes[pageID].nextNode->pageID;
	// 		// increase turn
	// 		atomicAdd(&d_LLturn, (unsigned long long)1);
	// 	}
	// 	grid.sync();
	// } while (d_LLturn<d_LLticket);
	// unsigned long long stop = (unsigned long long)clock();
	// if (stepCount) *stepCount = (int)(stop - start);
	// return 0;
}


// atomically get d_TailNodeID and replace with -1 until one thread replace it with a legit nodeID
// if successfully obtained the tail node (!=-1), replace it with the new tail(=pageID)
__device__ void freePageLinkedList(int pageID, int *stepCount){
	unsigned long long start = (unsigned long long)clock();
	volatile int nodeID = -1;
	volatile int pageID_ = pageID;

	while (nodeID==-1){
		nodeID = atomicExch((int*)&d_TailNodeID, -1);
		if (nodeID != -1){	// this thread successfully capture the tail node
			volatile Node_t *current_node = &d_nodes[pageID_];
			current_node->nextNode = NULL;	// probably redundant
			// now nodeID is the index to tail node
			volatile Node_t *tail_node = &d_nodes[nodeID];
			// update tailnode's next
			tail_node->nextNode = (Node_t*)current_node;
			// return new tail node (pageID) to d_TailNodeID
			atomicExch((int*)&d_TailNodeID, (int)pageID_);
		}
	}
	unsigned long long stop = (unsigned long long)clock();
	if (stepCount) *stepCount = (int)(stop - start);

	// FA MUTEX
	// cg::grid_group grid = cg::this_grid();
	// volatile unsigned long long thread_ticket;
	// thread_ticket = atomicAdd(&d_LLticket, (unsigned long long)1);
	// grid.sync();
	// // access linked list
	// do {
	// 	if (thread_ticket == d_LLturn){
	// 		Node_t *current_node = &d_nodes[pageID];
	// 		current_node->nextNode = NULL;
	// 		// update current tail's node
	// 		Node_t *tail_node = &d_nodes[d_TailNodeID];
	// 		tail_node->nextNode = (Node_t*)current_node;
	// 		// make current node tail
	// 		d_TailNodeID = pageID;
	// 		// increase turn
	// 		atomicAdd(&d_LLturn, (unsigned long long)1);
	// 	}
	// 	grid.sync();
	// } while (d_LLturn<d_LLticket);
}

void resetBufferLinkedList(){
	initPagesLinkedList_kernel <<< ceil((float)TOTAL_N_PAGES/32), 32 >>> ();
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
}

__global__ void printNumPagesLeftLinkedList_kernel(){
	if (d_HeadNodeID<0){
		printf("[LL info] no free page\n");
		return;
	}
	Node_t *node = &d_nodes[d_HeadNodeID];
	int count = 0;
	while (node){
		count++;
		node = node->nextNode;
	}
	printf("[LL info] Number of pages in linked list: %d, start=%d, end=%d \n", count, d_HeadNodeID, d_TailNodeID);
}

__host__ void printNumPagesLeftLinkedList(){
	printNumPagesLeftLinkedList_kernel <<< 1, 1 >>> ();
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
}



/* ------------------------------------- Single Clock Implementation -------------------------------------------- */
static __device__ int d_SingleClockArm;		// index to page number, from 0 to TOTAL_N_PAGES-1
static __device__ int *d_pageMapSingleClock;

/* initialize TOTAL_N_PAGES pages on GPU's global memory, each page is PAGE_SIZE large
also initialize the page map structure with all 0 (all free)
set clock arm to 0
 */
__host__ void initPagesSingleClock(){
	// initialize actual pages
	initPages();
	// initialize page map
	void *h_pageMapSingleClock;
	gpuErrchk( cudaMalloc((void**)&h_pageMapSingleClock, TOTAL_N_PAGES*sizeof(int)) );
	gpuErrchk( cudaMemcpyToSymbol(d_pageMapSingleClock, &h_pageMapSingleClock, sizeof(void*)) );
	gpuErrchk( cudaMemset(h_pageMapSingleClock, 0, TOTAL_N_PAGES*sizeof(int)) );
	// set clock arm to 0
	int zero = 0;
	gpuErrchk( cudaMemcpyToSymbol(d_SingleClockArm, &zero, sizeof(int)) );
}

__device__ int getPageSingleClock(int *step_count){
	bool foundFreePage = false;
	int pageID;
	int stepCount = 0;
	while (!foundFreePage){
		stepCount++;
		// atomically shift the arm
		pageID = atomicAdd(&d_SingleClockArm, 1);
		pageID = pageID % TOTAL_N_PAGES;
		// check if the obtained position is available
		if (d_pageMapSingleClock[pageID] == 0){
			foundFreePage = true;
			// mark page as used
			atomicExch(&d_pageMapSingleClock[pageID], 1);
		}
	}
	if (step_count) *step_count = stepCount;
	return pageID;
}

__device__ void freePageSingleClock(int pageID, int *step_count){
	atomicExch(&(d_pageMapSingleClock[pageID]), 0);
}

__global__ void printNumPagesLeftSingleClock_kernel(){
	int count = 0;
	for (int i=0; i<TOTAL_N_PAGES; i++){
		if (d_pageMapSingleClock[i]==0) count++;
	}
	printf("[SC info] Number of free pages: %d \n", count);
}
__host__ void printNumPagesLeftSingleClock(){
	printNumPagesLeftSingleClock_kernel <<< 1, 1 >>> ();
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
}

__global__ void resetBufferSingleClock_kernel(){
	memset(d_pageMapSingleClock, 0, TOTAL_N_PAGES*sizeof(int));
}
__host__ void resetBufferSingleClock(){
	resetBufferSingleClock_kernel <<< 1, 1 >>> ();
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
}

/* ------------------------------------- Parallel Clock Implementation -------------------------------------------- */
#define MAX_THREADS_DISTCLOCK 1000000
static __device__ int *d_DistClockArm;		// indexes to page number, from 0 to TOTAL_N_PAGES-1
static __device__ int *d_pageMapDistClock;

/* initialize TOTAL_N_PAGES pages on GPU's global memory, each page is PAGE_SIZE large
also initialize the page map structure with all 0 (all free)
randomize a number for each clock arm
 */
static __global__ void randomizeArmDistClock(){
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if (tid>=MAX_THREADS_DISTCLOCK) return;
	int16_t Clock = (int16_t)clock();
	int seed = (tid<<15) + Clock;
	seed = RNG_LCG(seed);
	seed = seed % TOTAL_N_PAGES;
	d_DistClockArm[tid] = seed;
}

__host__ void initPagesDistClock(){
	// initialize actual pages
	initPages();
	// initialize page map
	void *h_pageMapDistClock;
	gpuErrchk( cudaMalloc((void**)&h_pageMapDistClock, TOTAL_N_PAGES*sizeof(int)) );
	gpuErrchk( cudaMemcpyToSymbol(d_pageMapDistClock, &h_pageMapDistClock, sizeof(void*)) );
	gpuErrchk( cudaMemset(h_pageMapDistClock, 0, TOTAL_N_PAGES*sizeof(int)) );
	// randomize arms
	void *_d_DistClockArm;
	gpuErrchk( cudaMalloc((void**)&_d_DistClockArm, MAX_THREADS_DISTCLOCK*sizeof(int)) );
	gpuErrchk( cudaMemcpyToSymbol(d_DistClockArm, &_d_DistClockArm, sizeof(void*)) );
	randomizeArmDistClock <<< ceil((float)MAX_THREADS_DISTCLOCK/32), 32 >>> ();
}

__device__ int getPageDistClock(int *step_count){
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	bool foundFreePage = false;
	int pageID;
	int stepCount = 0;
	while (!foundFreePage){
		stepCount++;
		pageID = d_DistClockArm[tid]%TOTAL_N_PAGES;
		// check if page at current arm is free
		if (d_pageMapDistClock[pageID]==0){
			// try to mark page as taken
			int old_val = atomicExch(&(d_pageMapDistClock[pageID]), 1);
			if (old_val == 0)
				// this thread got the page
				foundFreePage = true;
		}
		// move arm
		d_DistClockArm[tid] = pageID + 1;
	}
	if (step_count) *step_count = stepCount;
	return pageID;
}

__device__ void freePageDistClock(int pageID, int *step_count){
	atomicExch(&(d_pageMapDistClock[pageID]), 0);
}

__global__ void printNumPagesLeftDistClock_kernel(){
	int count = 0;
	for (int i=0; i<TOTAL_N_PAGES; i++){
		if (d_pageMapDistClock[i]==0) count++;
	}
	printf("[DC info] Number of free pages: %d \n", count);
}
__host__ void printNumPagesLeftDistClock(){
	printNumPagesLeftDistClock_kernel <<< 1, 1 >>> ();
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
}

__global__ void resetBufferDistClock_kernel(){
	memset(d_pageMapDistClock, 0, TOTAL_N_PAGES*sizeof(int));
}
__host__ void resetBufferDistClock(){
	resetBufferDistClock_kernel <<< 1, 1 >>> ();
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
}



/* ------------- COLLABORATIVE RANDOM WALK IMPLEMENTATION --------------------- */

#define WARPSIZE 32

__device__ int *d_PageMapCollabRW;      // length TOTAL_N_PAGES

/* initialize TOTAL_N_PAGES pages on GPU's global memory, each page is PAGE_SIZE large
also initialize the page map structure with all 0 (all free) */
__host__ void initPagesCollabRW(){
	// initialize actual pages
	initPages();

	// initialize page map of length TOTAL_N_PAGES
	void *h_PageMapCollabRW;
	gpuErrchk( cudaMalloc((void**)&h_PageMapCollabRW, TOTAL_N_PAGES*sizeof(int)) );
	gpuErrchk( cudaMemcpyToSymbol(d_PageMapCollabRW, &h_PageMapCollabRW, sizeof(void*)) );
	gpuErrchk( cudaMemset(h_PageMapCollabRW, 0, TOTAL_N_PAGES*sizeof(int)) );
	
}

// if step_count is not null, write step count to it
__device__ int getPageCollabRW(int *stepCount){
	// initialize shared memory
	__shared__ int S_pageIDs_block[1024];  // this is to signal compiler to allocate enough for the entire block
	__shared__ int S_NPagesFound_blk[32];  
	int *S_pageIDs; 			// this is the shared array for the warp, calculated below
	int warpID = threadIdx.x>>5;
	S_pageIDs = &S_pageIDs_block[warpID<<5];
	int *S_NPagesFound = &S_NPagesFound_blk[warpID];    // keep track of number of pages found per warp
	if ((threadIdx.x%32)==0) *S_NPagesFound = 0;
	// find number of requests
	int NRequests = __popc(__activemask());

	// all active threads try find a random page every round
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	int seed = (tid<<15) + ((int16_t)clock());
	int stepCount_ = 0;
	// variable meaning:
		// S_NPagesFound: warp-shared counter of number of pages found 
	while (*S_NPagesFound<NRequests) {
		stepCount_++;
		// find a random page
		seed = RNG_LCG(seed);
		unsigned pageID_ = (unsigned)seed % TOTAL_N_PAGES;
		// check if page is available
		if (d_PageMapCollabRW[pageID_]==0){
			// if page is available, try to grab it
			int result = atomicExch(&d_PageMapCollabRW[pageID_],1);
			// if successful, add 1 to S_NPagesFound and add pageID to S_pageIDs
			if (result==0){
				int pos = atomicAdd(S_NPagesFound, 1);
				// if we have found more than what we need, return the page
				if (pos>=NRequests)
					freePageCollabRW(pageID_);
				// otherwise, add the pageID to the shared array
				else 
					S_pageIDs[pos] = pageID_;
			}
		}
	}

	// we have found enough pages and stored their IDs on shared memory
	// now grab them from shared memory
	int pos = atomicSub(S_NPagesFound, 1) - 1;
	int pageID = S_pageIDs[pos];

	if (stepCount) *stepCount = stepCount_;
	return pageID;
}


__device__ void freePageCollabRW(int pageID){
	atomicAnd(&d_PageMapCollabRW[pageID], 0);
}


__global__ void resetBufferCollabRW_kernel(){
	// set page map to 0
	memset(d_PageMapCollabRW, 0, TOTAL_N_PAGES*sizeof(int));
}

__host__ void resetBufferCollabRW(){
	resetBufferCollabRW_kernel <<< 1, 1 >>> ();
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
}

__global__ void printNumPagesLeftCollabRW_kernel(){
	int count = 0;
	for (int i=0; i<TOTAL_N_PAGES; i++){
		if (d_PageMapCollabRW[i]==0) count++;
	}
	printf("[CoRW info] Number of free pages: %d \n", count);
}

__host__ void printNumPagesLeftCollabRW(){
	printNumPagesLeftCollabRW_kernel <<< 1, 1 >>> ();
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
}


/* ------------- COLLABORATIVE RANDOM WALK IMPLEMENTATION WITH 32-BIT BITMAP --------------------- */

/* Arrange the N elements of ARRAY in random order.
   Only effective if N is much smaller than RAND_MAX;
   if this may not be the case, use a better random
   number generator. */
#define WARPSIZE 32
// an integer contains 32 bits corresponding to 32 pages

__device__ unsigned *d_pageMapCollabRW_BM;      // length TOTAL_N_PAGES/32
__device__ int *d_mapLockCollabRW_BM; 	   // length TOTAL_N_PAGES/32, 0 = free, 1 is locked

/* initialize TOTAL_N_PAGES pages on GPU's global memory, each page is PAGE_SIZE large
also initialize the page map structure with all 0 (all free) */
__host__ void initPagesCollabRW_BM(){
	// initialize actual pages
	initPages();

	// initialize page map of length TOTAL_N_PAGES
	void *h_pageMapCollabRW_BM;
	gpuErrchk( cudaMalloc((void**)&h_pageMapCollabRW_BM, TOTAL_N_PAGES) );
	gpuErrchk( cudaMemcpyToSymbol(d_pageMapCollabRW_BM, &h_pageMapCollabRW_BM, sizeof(void*)) );
	gpuErrchk( cudaMemset(h_pageMapCollabRW_BM, 0, TOTAL_N_PAGES) );
	// initialize map lock of length TOTAL_N_PAGES
	void *h_mapLockCollabRW_BM;
	gpuErrchk( cudaMalloc((void**)&h_mapLockCollabRW_BM, TOTAL_N_PAGES) );
	gpuErrchk( cudaMemcpyToSymbol(d_mapLockCollabRW_BM, &h_mapLockCollabRW_BM, sizeof(void*)) );
	gpuErrchk( cudaMemset(h_mapLockCollabRW_BM, 0, TOTAL_N_PAGES) );
}


// // helper function for getPageCollabRW_BM
// static inline __device__ unsigned generateMask(unsigned currentMask, int *NRequests){
// 	// currentMask: current value of 32-bit mask
// 	// return: a new mask for acquiring the 0 bits on currentMask, with up to NRequests 1-bits
// 	//         Nrequests: number of remaining Requests after doing this
// 	unsigned newMask = 0;
// 	for (int i=0; i<32; i++){
// 		if ((*NRequests)==0) break;  // stop when we have acquired NRequests bits
// 		if ((currentMask&(1<<i))==0){ // if this bit is 0, acquire it
// 			newMask|=(1<<i);
// 			(*NRequests)--;
// 		}
// 	}
// 	return newMask;
// }

// // helper function for getPageCollabRW_BM
// // return the max value of x in the warp
// static inline int __device__ warpReduceMax(int x){
// 	int max = x;
// 	int tmp;
// 	for (int mask = WARPSIZE>>1; mask>0; mask>>=1){
//     	tmp = __shfl_xor_sync(0xffffffff, max, mask);
//     	max = tmp>max ? tmp : max;
// 	}
//   	return max;
// }

/* getPageCollabRW_BM:
    - Calculate NRequests: number of threads calling this function in a warp
    - page status (free/taken) is stored as 0/1 bits on global memory
        this is an array of BITMAP_LENGTH=TOTAL_N_PAGES/32 integers on global memory 
    - In this function, threads in a warp try to obtain pages together
    - while NumPagesFound_SharedMem < NRequests:
        - All threads read a random chunk of consecutive 1024 bits (32 integers)
        - All threads do:
        	- Try to lock the integer
        	- If the thread obtained the lock, do:
            	- If the integer != 0xffffffff (all 32 bits are 1):
		            - Count the number of 0-bits on the integer (assign to N0Bits)
		            - pagesFound_old = atomicAdd(NumPagesFound_SharedMem, N0Bits)
		            - If pagesFound_old>NRequests:
		            	do nothing, we don't need the integer in this thread to fill up NRequests for the warp
		            - Else If pagesFound_old+N0Bits<=NRequests:
		            	we can flip all the 0 on the integer in this thread
					- Else:
						In this case, we have to flip (pagesFound_old+N0Bits-NRequests) 0-bits on the integer in this thread
					- Flip the bits as we determined above
					- If this thread flipped some bits, calculate their corresponding pageID and write to a shared mem array
    			- Release lock
    - Assignment of PageID:
        - atomicAdd to an integer on shared mem to get a position
        - pageID = sharedMem[pos]
 */


#define BITMAP_LENGTH (TOTAL_N_PAGES/32)

static inline __device__ int calculateFlipMask(int groupValue, int NBits){
	// return a mask that would flip NBits 0-bits on groupValue to 1
	// helper function for getPageCollabRW_BM
	int flipMask = 0;
	int count = 0;
	for (int i=0; i<32; i++){	// loop thru bits
		if (count==NBits)
			break;
		if ((groupValue&(1<<i))==0){
			flipMask |= (1<<i);
			count++;
		}
	}
	return flipMask;
}

__device__ int getPageCollabRW_BM(int *stepCount){
	// initialize shared memory for the entire block
	__shared__ int S_pageIDs_block[1024];	// this is to signal compiler to allocate enough for the entire block
	__shared__ int S_pagesFound_block[32];	// this is to signal compiler to allocate enough for the entire block
	// initialize shared memory for this warp
	int warpID = threadIdx.x>>5;
	int *S_pageIDs = &S_pageIDs_block[warpID<<5]; // this is the shared array for the warp, calculated below
												  // S_pageIDs will store the pageIDs acquired, to be distributed later
	int *S_pagesFound = &S_pagesFound_block[warpID];    // store the number of pages this warp has found

	// find number of requests
	int activeMask = __activemask();
	int NRequests = __popc(activeMask);
	int laneID = threadIdx.x%32;  // threadID within a warp
	// find ID of one active lane, for actions such as setting sharedMem to 0
	int firstLaneID = __ffs(activeMask) - 1;

	// all threads in this warp randomize to the same position
	// we can randomize to the same position by setting the same seed across warp
	if (laneID==firstLaneID)
		*S_pagesFound = 0;  // keep track of the number of pages this warp has found together
	// prepare seed for RNG
	int seed;
	// prepare seed in one lane, then broadcast
	if (laneID==firstLaneID)
		seed = ((blockIdx.x*blockDim.x+threadIdx.x)<<15) + ((int16_t)clock());
	__syncwarp();
	seed = __shfl_sync(activeMask, seed, firstLaneID);
	int stepCount_ = 0;
	while (*S_pagesFound<NRequests) {
		stepCount_++;
		// All threads read a random chunk of consecutive 1024 bits (32 integers)
		seed = RNG_LCG(seed);
		// force groupID to be between 0 and BITMAP_LENGTH-32
		unsigned groupID = (unsigned)seed % (BITMAP_LENGTH-32);
		// increment by laneID to get the position of the integer this thread will read
		groupID += laneID;
// if (blockIdx.x==0) printf("t %d, s %d\n", threadIdx.x, groupID);
		if (d_pageMapCollabRW_BM[groupID]!=0xffffffff){
			// Try to lock the integer
			volatile int lock = atomicExch(&d_mapLockCollabRW_BM[groupID], 1);
			if (!lock){ // we got the lock
				// read the integer
				int groupValue = d_pageMapCollabRW_BM[groupID];
				if (groupValue != 0xffffffff){	// all 1's means we can't do anything
					// Count the number of 0-bits on the integer
					int N0Bits = __popc(~groupValue);
					// add this number to the shared memory
					int pagesFound_old = atomicAdd(S_pagesFound, N0Bits);
					// determine which bits to flip
					int flipMask;
					if (pagesFound_old>=NRequests)
						flipMask = 0;	// don't flip any bit because we already found enough pages
					else if (pagesFound_old+N0Bits<=NRequests){
						flipMask = calculateFlipMask(groupValue, N0Bits); // flip all N0Bits bits
					}
					else  // in this case, we need to flip (pagesFound_old+N0Bits-NRequests) 0-bits
						flipMask = calculateFlipMask(groupValue, NRequests - pagesFound_old);
					// Flip the bits as we determined above
					if (flipMask!=0){
						atomicOr(&d_pageMapCollabRW_BM[groupID], flipMask);
					}
					// calculate corresponding pageIDs and save to shared mem array
					if (flipMask!=0){
						int pageOffset = groupID*32;
						int arrayPos = pagesFound_old;	// position on the shared mem array to write
						for (int i=0; i<32; i++){	// loop thru bits
							if (flipMask&(1<<i)){
								int pageID = pageOffset + i;
								S_pageIDs[arrayPos] = pageID;
								arrayPos++;
							}
						}
					}
				}
			}
			// release the lock
			if (!lock)
				atomicExch(&d_mapLockCollabRW_BM[groupID], 0);
		}
		__syncwarp();
	}

	// we have found enough pages and stored their IDs on shared memory
	// Assignment of PageID to threads
	int *S_Requests = S_pagesFound;	// reuse for positioning
	if (laneID == firstLaneID)
		*S_Requests = 0;
	__syncwarp();
	int pos = atomicAdd(S_Requests, 1);
	int pageID = S_pageIDs[pos];

	if (stepCount) *stepCount = stepCount_;
	return pageID;
}

/*
    - compute groupID = pageID/32
    - compute bit position = pageID - groupID*32
    - navigate to groupID location and flip the bit position to 0
 */
__device__ void freePageCollabRW_BM(int pageID){
	int groupID = pageID/32;
	int bitPosition = pageID - groupID*32;
	volatile int lock = 1;
	while (lock){
		// try to grab lock
		lock = atomicExch(&d_mapLockCollabRW_BM[groupID], 1);
		if (!lock) // got the lock
			d_pageMapCollabRW_BM[groupID] &= ~(1<<bitPosition);
		if (!lock) // release lock
			atomicExch(&d_mapLockCollabRW_BM[groupID], 0);
	}
}


__global__ void resetBufferCollabRW_BM_kernel(){
	// set page map to 0
	memset(d_pageMapCollabRW_BM, 0, BITMAP_LENGTH*sizeof(int));
}

__host__ void resetBufferCollabRW_BM(){
	resetBufferCollabRW_BM_kernel <<< 1, 1 >>> ();
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
}

__global__ void printNumPagesLeftCollabRW_BM_kernel(){
	int count = 0;
	for (int i=0; i<BITMAP_LENGTH; i++)
		count += __popc(~d_pageMapCollabRW_BM[i]);
	printf("[CoRW_BM info] Number of free pages: %d \n", count);
}

__host__ void printNumPagesLeftCollabRW_BM(){
	printNumPagesLeftCollabRW_BM_kernel <<< 1, 1 >>> ();
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
}


/* --------------------- RW + BitMap IMPLEMENTATION -------------------------------------*/
__device__ int *d_PageMapRandomWalk_BM;  // length TOTAL_N_PAGES/32
__device__ int *d_RandomWalk_BM_Lock;    // length TOTAL_N_PAGES/32
/* initialize TOTAL_N_PAGES pages on GPU's global memory, each page is PAGE_SIZE large
also initialize the page map structure with all 0 (all free) */
__host__ void initPagesRandomWalk_BM(){
	// initialize actual pages
	initPages();

	// initialize page map
	void *h_PageMapRandomWalk_BM;
	gpuErrchk( cudaMalloc((void**)&h_PageMapRandomWalk_BM, TOTAL_N_PAGES) );
	gpuErrchk( cudaMemcpyToSymbol(d_PageMapRandomWalk_BM, &h_PageMapRandomWalk_BM, sizeof(void*)) );
	gpuErrchk( cudaMemset(h_PageMapRandomWalk_BM, 0, TOTAL_N_PAGES) );

	// initialize locks
	void *h_RandomWalk_BM_Lock;
	gpuErrchk( cudaMalloc((void**)&h_RandomWalk_BM_Lock, TOTAL_N_PAGES) );
	gpuErrchk( cudaMemcpyToSymbol(d_RandomWalk_BM_Lock, &h_RandomWalk_BM_Lock, sizeof(void*)) );
	gpuErrchk( cudaMemset(h_RandomWalk_BM_Lock, 0, TOTAL_N_PAGES) );
}



// if step_count is not null, write step count to it
// BITMAP_LENGTH = 32
__device__ int getPageRandomWalk_BM(int *stepCount){
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	int16_t Clock = (int16_t)clock();
	int seed = (tid<<15) + Clock;
	// randomize pages and try to grab a page
	int step_count = 0;
	int pageID;
	bool foundPage = false;
	while(!foundPage){
		step_count++;
		// perform random jump
		seed = RNG_LCG(seed);
		unsigned groupID = (unsigned)seed % BITMAP_LENGTH;
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
				foundPage = true;
				break; // break the inner while loop
			}
		}
	}

	// int pageID = tid*100;
	// atomicExch(&d_PageMapRandomWalk[pageID],1);
	// int step_count = 1;

	if (stepCount) *stepCount = step_count;
	return pageID;
}

// In this implementation, X is at most 32
__device__ int getXPageRandomWalk_BM(int X, int *stepCount){
	if (X<1 || X>32){printf("invalid number of page requested \n"); __trap();}
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	int16_t Clock = (int16_t)clock();
	int seed = (tid<<15) + Clock;
	// randomize pages and try to grab X pages
	int step_count = 0;
	int pageID;
	bool foundPage = false;
	int bitMaskXBits = ~(0xffffffff<<X); // a mask of X 1-bits on the least significant side and 0s on the most significant side
	while(!foundPage){
		step_count++;
		// perform random jump
		seed = RNG_LCG(seed);
		unsigned groupID = (unsigned)seed % BITMAP_LENGTH;
		while (__popc(~d_PageMapRandomWalk_BM[groupID])>=X){	// keep trying to get pages while number of 0-bits is >=X
			int groupValue = d_PageMapRandomWalk_BM[groupID];
			// try to flip X consecutive bits from 0 to 1
			int bitPosition = __ffs(~groupValue) - 1; // 0-based index
			int bitMaskXBits_shifted = bitMaskXBits<<bitPosition;
			int result = atomicOr(&d_PageMapRandomWalk_BM[groupID], bitMaskXBits_shifted);
			// check if we actually flipped the bit
			if ((result & bitMaskXBits_shifted) == 0){
				// we flipped the bit
				// calculate the corresponding first pageID of these X consecutive pages
				pageID = groupID<<5 + bitPosition;
				foundPage = true;
				break; // break the inner while loop
			} else {
				// if not (due to thread collisions), revert changes
				int revertBitMask = ~(result&bitMaskXBits_shifted);
				atomicAnd(&d_PageMapRandomWalk_BM[groupID], revertBitMask);
			}
		}
	}

	if (stepCount) *stepCount = step_count;
	return pageID;
}


__device__ void freePageRandomWalk_BM(int pageID){
	int groupID = pageID>>5;
	int bitPosition = pageID - (groupID<<5);
	volatile int lock = 1;
	while (lock){
		// try to grab lock
		lock = atomicExch(&d_RandomWalk_BM_Lock[groupID], 1);
		if (!lock) // got the lock
			d_PageMapRandomWalk_BM[groupID] &= ~(1<<bitPosition);
		if (!lock) // release lock
			atomicExch(&d_RandomWalk_BM_Lock[groupID], 0);
	}
}


__global__ void resetBufferRandomWalk_BM_kernel(){
	memset(d_PageMapRandomWalk_BM, 0, TOTAL_N_PAGES);
}

__host__ void resetBufferRandomWalk_BM(){
	resetBufferRandomWalk_BM_kernel <<< 1, 1 >>> ();
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
}

__global__ void printNumPagesLeftRandomWalk_BM_kernel(){
	int count = 0;
	for (int i=0; i<BITMAP_LENGTH; i++){
		count += __popc(~d_PageMapRandomWalk_BM[i]);
	}
	printf("[RW info] Number of free pages: %d \n", count);
}

__host__ void printNumPagesLeftRandomWalk_BM(){
	printNumPagesLeftRandomWalk_BM_kernel <<< 1, 1 >>> ();
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
}

/* --------------------- CRW + BitMap IMPLEMENTATION -------------------------------------*/
__device__ int *d_CRW_BM_LastGroupID;	// length 65536*1024
/* initialize TOTAL_N_PAGES pages on GPU's global memory, each page is PAGE_SIZE large
also initialize the page map structure with all 0 (all free) */
__host__ void initPagesClusteredRandomWalk_BM(){
	initPagesRandomWalk_BM();

	// initialize global array to store last group visited
	void *h_CRW_BM_LastGroupID;
	gpuErrchk( cudaMalloc((void**)&h_CRW_BM_LastGroupID, 65536*1024*sizeof(int)) );
	gpuErrchk( cudaMemcpyToSymbol(d_CRW_BM_LastGroupID, &h_CRW_BM_LastGroupID, sizeof(void*)) );
	gpuErrchk( cudaMemset(h_CRW_BM_LastGroupID, -1, 65536*1024*sizeof(int)) );
}



// if step_count is not null, write step count to it
// BITMAP_LENGTH = (TOTAL_N_PAGES/32)
// try to lock a group before modifying it
__device__ int getPageClusteredRandomWalk_BM(int *stepCount){
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	int16_t Clock = (int16_t)clock();
	int seed = (tid<<15) + Clock;
	int pageID;
	int groupID = d_CRW_BM_LastGroupID[tid];
	bool foundPage = false;

	// retrieve the last groupID visited
	if (groupID==-1){
		// initialize the first random groupID
		seed = (RNG_LCG(seed) % 3) + (int)(tid*3.1245);
		groupID = (unsigned)seed % BITMAP_LENGTH;
	}
// printf("tid %d seed %d group %d\n", tid, seed, groupID);
	// check if last group visited still has space, if not, go to next group
	if (d_PageMapRandomWalk_BM[groupID]==0xffffffff){
		groupID++;
		if (groupID==BITMAP_LENGTH) groupID = 0;
	}
// if (tid==0) printf("%d %d\n", groupID, d_PageMapRandomWalk_BM[groupID]);
	while (!foundPage && d_PageMapRandomWalk_BM[groupID]!=0xffffffff){
		int groupValue = d_PageMapRandomWalk_BM[groupID];
		// try to flip one of the 0-bit to 1
		int bitPosition = __ffs(~groupValue) - 1;
		int result = atomicOr(&d_PageMapRandomWalk_BM[groupID], 1<<bitPosition);
// if (tid==0)printf("---- %d, %d, %d\n", groupID, bitPosition, result & (1<<bitPosition));
		// check if we actually flipped the bit
		if ((result & (1<<bitPosition)) == 0){
			// we flipped the bit
			// calculate the corresponding pageID
			pageID = groupID*32 + bitPosition;
			foundPage = true;
		}
	}
// printf("%u, %d\n", groupID, __ffs(d_PageMapRandomWalk_BM[groupID]));

	int step_count = 1;
	// perform random jump
	while(!foundPage){
		step_count++;
		seed = RNG_LCG(seed);
		groupID = seed % BITMAP_LENGTH;
		if (groupID<0) groupID = -groupID;
		while (!foundPage && d_PageMapRandomWalk_BM[groupID]!=0xffffffff){
			int groupValue = d_PageMapRandomWalk_BM[groupID];
			// try to flip one of the 0-bit to 1
			int bitPosition = __ffs(~groupValue) - 1;
			int result = atomicOr(&d_PageMapRandomWalk_BM[groupID], 1<<bitPosition);
			// check if we actually flipped the bit
			if ((result & (1<<bitPosition)) == 0){
				// we flipped the bit
				// calculate the corresponding pageID
				pageID = groupID*32 + bitPosition;
				foundPage = true;
// if (tid==0)printf("-- case 2 %d, %d\n", groupID, bitPosition);
			}
		}
	}
	// write last groupID visited
	d_CRW_BM_LastGroupID[tid] = groupID;
// if (tid==0)printf("---- write %d\n", d_CRW_BM_LastGroupID[tid]);

	if (stepCount) *stepCount = step_count;
	return pageID;
}


__device__ void freePageClusteredRandomWalk_BM(int pageID){
	int groupID = pageID>>5;
	int bitPosition = pageID - (groupID<<5);
	volatile int lock = 1;
	while (lock){
		// try to grab lock
		lock = atomicExch(&d_RandomWalk_BM_Lock[groupID], 1);
		if (!lock) // got the lock
			d_PageMapRandomWalk_BM[groupID] &= ~(1<<bitPosition);
		if (!lock) // release lock
			atomicExch(&d_RandomWalk_BM_Lock[groupID], 0);
	}
}


__global__ void resetBufferClusteredRandomWalk_BM_kernel(){
	memset(d_PageMapRandomWalk_BM, 0, TOTAL_N_PAGES);
	memset(d_CRW_BM_LastGroupID, -1, BITMAP_LENGTH*sizeof(int));
}

__host__ void resetBufferClusteredRandomWalk_BM(){
	resetBufferClusteredRandomWalk_BM_kernel <<< 1, 1 >>> ();
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
}

__global__ void printNumPagesLeftClusteredRandomWalk_BM_kernel(){
	int count = 0;
	for (int i=0; i<BITMAP_LENGTH; i++){
		count += __popc(~d_PageMapRandomWalk_BM[i]);
	}
	printf("[CRW_BM info] Number of free pages: %d \n", count);
//count = 0;
//for (int i=0; i<10000; i++){
//	int groupID = d_CRW_BM_LastGroupID[i];
//	int groupID1 = groupID+1;
//	if (groupID==-1) continue;
	// if (groupID1==BITMAP_LENGTH) groupID = 0;
	// if (d_PageMapRandomWalk_BM[groupID]==0xffffffff && d_PageMapRandomWalk_BM[groupID1]==0xffffffff)
	// 	count++;
//	printf("%d,%d: %d %d\n", i, groupID, __popc(~d_PageMapRandomWalk_BM[groupID]), __popc(~d_PageMapRandomWalk_BM[groupID]));
//}
// printf("%d\n", count);
}

__host__ void printNumPagesLeftClusteredRandomWalk_BM(){
	printNumPagesLeftClusteredRandomWalk_BM_kernel <<< 1, 1 >>> ();
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
}
