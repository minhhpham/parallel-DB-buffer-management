#include "parallelPage.cuh"
#include <curand_kernel.h>

/* actual pages for all strategies */
__device__ void *d_pages;					// actual address to pages
__device__ void *pageAddress(int pageID){
	return (void*)((char*)d_pages + pageID*PAGE_SIZE);
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
static __device__ Node_t *d_HeadNode;	// pointer to first free node
static __device__ int d_lockHeadNode;	// 0 means free
static __device__ Node_t *d_TailNode;	// pointer to last free node
static __device__ int d_lockTailNode;	// 0 means free

/* RANDOM WALK IMPLEMENTATION */
/* initialize TOTAL_N_PAGES pages on GPU's global memory, each page is PAGE_SIZE large
also initialize the page map structure with all 0 (all free) */
__host__ void initPagesRandomWalk(){
	// initialize pages, allocate a big chunk for all pages
	void *h_pages;
	gpuErrchk( cudaMalloc((void**)&h_pages, (size_t)TOTAL_N_PAGES*PAGE_SIZE) );
	gpuErrchk( cudaMemcpyToSymbol(d_pages, &h_pages, sizeof(void*)) );

	// initialize page map
	void *h_PageMapRandomWalk;
	gpuErrchk( cudaMalloc((void**)&h_PageMapRandomWalk, TOTAL_N_PAGES*sizeof(int)) );
	gpuErrchk( cudaMemcpyToSymbol(d_PageMapRandomWalk, &h_PageMapRandomWalk, sizeof(void*)) );
	gpuErrchk( cudaMemset(h_PageMapRandomWalk, 0, TOTAL_N_PAGES*sizeof(int)) );
}


// if step_count is not null, write step count to it
__device__ int getPageRandomWalk(int *stepCount){
	curandState_t state;
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	curand_init(clock(), // seed
			  tid,	// sequence number
			  0,	// offset
			  &state);
	// randomize pages and try
	int pageID = curand(&state) % TOTAL_N_PAGES;
	if (pageID<0) pageID = -pageID;
	int step_count = 1;
	while(atomicExch(&d_PageMapRandomWalk[pageID],1) == 1){
		pageID = curand(&state) % TOTAL_N_PAGES;
		step_count++;
	}
	if (stepCount) *stepCount = step_count;
	return pageID;
}

__device__ void freePageRandomWalk(int pageID){
	atomicExch(&d_PageMapRandomWalk[pageID], 0);
}


__global__ void printNumPagesLeftRandomWalk_kernel(){
	int count = 0;
	for (int i=0; i<TOTAL_N_PAGES; i++){
		if (d_PageMapRandomWalk[i]==0) count++;
	}
	printf("Number of free pages: %d \n", count);
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
	gpuErrchk( cudaMalloc((void**)&d_tmp, TOTAL_N_PAGES*sizeof(int)) );
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

__host__ void printNumPagesLeftClusteredRandomWalk(){
	printNumPagesLeftRandomWalk();
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
		d_HeadNode = &d_nodes[0];	// first node is head
	// set tail node
	if (i==TOTAL_N_PAGES-1)			// last node is tail
		d_TailNode = &d_nodes[TOTAL_N_PAGES-1];
	// set head and tail locks free
	if (i==0){
		d_lockHeadNode = 0;
		d_lockTailNode = 0;
	}
}

__host__ void initPagesLinkedList(){
	// initialize pages, allocate a big chunk for all pages
	void *h_pages;
	gpuErrchk( cudaMalloc((void**)&h_pages, (size_t)TOTAL_N_PAGES*PAGE_SIZE) );
	gpuErrchk( cudaMemcpyToSymbol(d_pages, &h_pages, sizeof(void*)) );

	// initialize linked list 
	Node_t *h_d_nodes;	// allocate nodes array and get pointer value on host, then copy this value to d_nodes
	gpuErrchk( cudaMalloc((void**)&h_d_nodes, TOTAL_N_PAGES*sizeof(Node_t)) );
	gpuErrchk( cudaMemcpyToSymbol(d_nodes, &h_d_nodes, sizeof(Node_t*)) );

	initPagesLinkedList_kernel <<< ceil((float)TOTAL_N_PAGES/32), 32 >>> ();
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
}

// if step_count is not null, write step count to it
__device__ int getPageLinkedList(int *stepCount){
	// obtain lock to modify head node
	int locked = 1;
	int pageID;
	int step_count = 0;
	unsigned int wait_time_ns = 32;	// wait 32 ns before trying to get lock again
	while (locked){
		locked = atomicExch(&d_lockHeadNode, 1);
		if (locked==0){
			// got lock access
			// check if OOM
			if (!d_HeadNode){printf("Out of pages!\n"); __trap();}
			pageID = d_HeadNode->pageID;
			// change head pointer
			Node_t *next_head = d_HeadNode->nextNode;
			atomicExch((unsigned long long*)&d_HeadNode, (unsigned long long)next_head);
			// d_HeadNode = d_HeadNode->nextNode;
			// invalidate obtained page's next
			d_nodes[pageID].nextNode = NULL;
			// release lock
			atomicExch(&d_lockHeadNode, 0) ;
		} else {
			__nanosleep(wait_time_ns);
			if (wait_time_ns < 256) wait_time_ns = wait_time_ns<<1;
		}
		step_count++;
	}
	if (stepCount) *stepCount = step_count;
	return pageID;
}

__device__ void freePageLinkedList(int pageID){
	// obtain lock to modify tail node
	int locked = 1;
	unsigned int wait_time_ns = 32;	// wait 32 ns before trying to get lock again
	while (locked){
		locked = atomicExch(&d_lockTailNode, 1);
		if (locked==0){
			// got lock access
			Node_t *new_node = &d_nodes[pageID];	// get the node from pre-allocated nodes
			new_node->pageID = pageID;	// this is probably not neccessary now
			new_node->nextNode = NULL;	// make this new node the tail node
			// modify current tail node to point to this new node
			if (d_TailNode){
				int tailNodeID = d_TailNode->pageID;
				d_nodes[tailNodeID].nextNode = new_node;
			}
			// make this new node the tail node
			atomicExch((unsigned long long*)&d_TailNode, (unsigned long long)new_node);
			// release lock
			atomicExch(&d_lockTailNode, 0);
		} else {
			__nanosleep(wait_time_ns);
			if (wait_time_ns < 256) wait_time_ns = wait_time_ns<<1;
		}
	}
}

__global__ void printNumPagesLeftLinkedList_kernel(){
	Node_t *node = d_HeadNode;
	int count = 0;
	if (node == 0){
		printf("no free page\n");
		return;
	}
	while (node){
		count++;
		node = node->nextNode;
	}
	printf("Number of pages in linked list: %d, start=%d, end=%d \n", count, d_HeadNode->pageID, d_TailNode->pageID);
}

__host__ void printNumPagesLeftLinkedList(){
	printNumPagesLeftLinkedList_kernel <<< 1, 1 >>> ();
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
}