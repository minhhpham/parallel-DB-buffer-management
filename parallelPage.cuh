#include <stdio.h>

#define TOTAL_N_PAGES 	1000000
#define PAGE_SIZE		4096		// each page is 4 KB

// function to return pointer to start of a page given pageID
extern __device__ void *pageAddress(int pageID); 

/* functions for Random Walk*/
extern __host__ void initPagesRandomWalk(); 		// must run before any kernel that has the 2 functions below
extern __device__ int getPageRandomWalk(int *step_count=0);
extern __device__ void freePageRandomWalk(int pageID);
extern __host__ void printNumPagesLeftRandomWalk();
extern __host__ void resetBufferRandomWalk();

/* functions for Clustered Random Walks */
extern __host__ void initPagesClusteredRandomWalk();	// must run before any kernel that has the 2 functions below
extern __device__ int getPageClusteredRandomWalk(int *step_count=0);
extern __device__ void freePageClusteredRandomWalk(int pageID);
extern __host__ void printNumPagesLeftClusteredRandomWalk();

/* functions for LinkedList */
extern __host__ void initPagesLinkedList();		// must run before any kernel that has the 2 functions below
extern __device__ int getPageLinkedList(int *step_count=0);
extern __device__ void freePageLinkedList(int pageID, int *step_count=0);
extern __host__ void printNumPagesLeftLinkedList();
extern __host__ void resetBufferLinkedList();


/*CUDA ERROR HANDLER*/
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
