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

/* functions for Single Clock */
extern __host__ void initPagesSingleClock();		// must run before any kernel that has the 2 functions below
extern __device__ int getPageSingleClock(int *step_count=0);
extern __device__ void freePageSingleClock(int pageID, int *step_count=0);
extern __host__ void printNumPagesLeftSingleClock();
extern __host__ void resetBufferSingleClock();

/* functions for Distributed Clock*/
extern __host__ void initPagesDistClock();		// must run before any kernel that has the 2 functions below
extern __device__ int getPageDistClock(int *step_count=0);
extern __device__ void freePageDistClock(int pageID, int *step_count=0);
extern __host__ void printNumPagesLeftDistClock();
extern __host__ void resetBufferDistClock();

/* functions for Collaborative Random Walk*/
extern __host__ void initPagesCollabRW();     // must run before any kernel that has the 2 functions below
extern __device__ int getPageCollabRW(int *step_count=0);
extern __device__ void freePageCollabRW(int pageID);
extern __host__ void printNumPagesLeftCollabRW();
extern __host__ void resetBufferCollabRW();

/* functions for Collaborative Random Walk + BitMap */
extern __host__ void initPagesCollabRW_BM();     // must run before any kernel that has the 2 functions below
extern __device__ int getPageCollabRW_BM(int *step_count=0);
extern __device__ void freePageCollabRW_BM(int pageID);
extern __host__ void printNumPagesLeftCollabRW_BM();
extern __host__ void resetBufferCollabRW_BM();

/* functions for Random Walk + BitMap */
extern __host__ void initPagesRandomWalk_BM();     // must run before any kernel that has the 2 functions below
extern __device__ int getPageRandomWalk_BM(int *step_count=0);
extern __device__ int getXPageRandomWalk_BM(int X, int *step_count=0);	// get X consecutive pages
extern __device__ void freePageRandomWalk_BM(int pageID);
extern __host__ void printNumPagesLeftRandomWalk_BM();
extern __host__ void resetBufferRandomWalk_BM();

/* functions for Clustered Random Walk + BitMap */
extern __host__ void initPagesClusteredRandomWalk_BM();     // must run before any kernel that has the 2 functions below
extern __device__ int getPageClusteredRandomWalk_BM(int *step_count=0);
extern __device__ int getXPageClusteredRandomWalk_BM(int X, int *step_count=0); // get X consecutive pages
extern __device__ void freePageClusteredRandomWalk_BM(int pageID);
extern __host__ void printNumPagesLeftClusteredRandomWalk_BM();
extern __host__ void resetBufferClusteredRandomWalk_BM();


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
