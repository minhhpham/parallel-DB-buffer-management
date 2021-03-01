#include "parallelPage.cuh"
#include <stdint.h>
#include <cooperative_groups.h>
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
}


/* function to replace curand, better efficiency */
#define LCG_M 1<<31
#define LCG_A 1103515245
#define LCG_C 12345
__device__ static inline int RNG_LCG(int seed){
    long long seed_ = (long long)seed;
    return (int)((LCG_A*seed_ + LCG_C)%(LCG_M));
}

// if step_count is not null, write step count to it
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