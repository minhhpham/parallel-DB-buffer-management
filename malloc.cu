#include "parallelPage.cuh"
#include "malloc.cuh"

#define INITBM()    initPagesRandomWalk_BM()	// buffer initializer
#define GETXPAGE(x) getXPageRandomWalk_BM(x);	// get page function from the buffer manager
#define FREEPAGE(pageID) freePageRandomWalk_BM(pageID)

/* CLASS GENERAL MEMORY MANAGER */
__host__ MemoryManager::MemoryManager(int _maxNThreads){
	m_maxNThreads = _maxNThreads;
	// initialize paging buffer manager
	INITBM();
}

__host__ MemoryManager::~MemoryManager(){
}

/*
 SIMPLE MALLOC:
	Main idea: Each thread holds a buffer chunk (multiple consecutive pages) for subsequent malloc calls. 
	When a chunk runs out, request a new buffer chunk with number of page = ceil(request_size/PAGE_SIZE)
	Store the allocated amount on the first 8 bytes of each chunk:
			4-bytes: |        0       |          1           |
			info:    | allocated size | Chunk available size |
			NOTE: Chunk available size = PAGE_SIZE * NPages
			NOTE: allocated size includes the first 8 bytes
	
 */

/* CONSTRUCTOR */
__host__ KernelMallocSimple::KernelMallocSimple(int _maxNThreads) : MemoryManager(_maxNThreads)
{
	m_maxNThreads = _maxNThreads;
	// initialize d_current MallocPage for each thread
	gpuErrchk( cudaMalloc((void**)&d_currentMallocPage, m_maxNThreads*sizeof(int)) );
	// memset to -1 
	gpuErrchk( cudaMemset(d_currentMallocPage, -1, m_maxNThreads*sizeof(int)) );
	
}
// end of constructor for KernelMallocSimple

/* DESTRUCTOR */
__host__ KernelMallocSimple::~KernelMallocSimple()
{
	// free paging buffer manager
	// TODO
	// free d_current MallocPage
	// gpuErrchk( cudaFree(d_currentMallocPage) );
}

/* MALLOC FUNCTION */
#define CHUNK_METADATA_SIZE 8   // first 8 bytes hold meta data
__device__ void* KernelMallocSimple::malloc(int size){
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	void *outPtr;
	// retrieve the page we are currently using for malloc
	int pageID = d_currentMallocPage[tid];

	bool need_new_chunk = false;
	// allocate new page if this has not been initialized or current chunk ran out, need to allocate a new chunk
	if (pageID==-1) need_new_chunk = true;
	else {
		// go to actual chunk
		void *page = pageAddress(pageID);
		int *allocated_size = (int*)page;  // amount of allocated size on first 4 bytes of chunk
		// check if the remaining size is enough for the request
		if (PAGE_SIZE-CHUNK_METADATA_SIZE-*allocated_size < size)
			need_new_chunk = true;
	}

	if (need_new_chunk){
		// calculate number of pages needed in this chunk
		int NPage = ceil((float)(size+CHUNK_METADATA_SIZE)/PAGE_SIZE);
		pageID = GETXPAGE(NPage);	// consecutive page allocation, ID of first page in the chunk
		void *page = pageAddress(pageID);
		// set first 4 bytes on the chunk (allocated size) to requested size
		int *allocated_size = (int*)page;
		*allocated_size = size + CHUNK_METADATA_SIZE;
		// set chunk size
		int *chunk_size = allocated_size + 1;
		*chunk_size = PAGE_SIZE*NPage;
		// calculate resulted ptr
		outPtr = (void*)((char*)pageAddress(pageID) + 8);
	} else {
		// just calculate the output pointer and increase allocated_size
		void *page = pageAddress(pageID);
		int *allocated_size = (int*)page;
		outPtr = (void*)((char*)page + (*allocated_size));
		(*allocated_size) += size;
	}
	
	// save pageID
	d_currentMallocPage[tid] = pageID;
	return outPtr;
}
// end of malloc function
/* FREE FUNCTION */

#undef CHUNK_METADATA_SIZE

