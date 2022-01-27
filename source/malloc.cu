#include "parallelPage.cuh"
#include "malloc.cuh"

// #define INITBM()    initPagesRandomWalk_BM()	// buffer initializer
// #define RESETBM()   resetBufferRandomWalk_BM()		// reset buffer
// #define GETXPAGE(x) getXPageRandomWalk_BM(x)	// get page function from the buffer manager
// #define FREEPAGE(pageID) freePageRandomWalk_BM(pageID)

#define INITBM()    initPagesCollabRW_BM()	// buffer initializer
#define RESETBM()   resetBufferCollabRW_BM()		// reset buffer
#define GETXPAGE(x) getXPageCollabRW_BM(x)	// get page function from the buffer manager
#define FREEPAGE(pageID) freePageCollabRW_BM(pageID)

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
	
	Chunk MetaData type (Store the allocated amount on the first bytes of each chunk):
			chunk available size: int, for allocating spaces
			chunk allocated size: int
			lastSpaceOffset: offset (in bytes) to spaceMeta of the last space (does not include space for chunk meta )
			NOTE: Chunk available size = PAGE_SIZE * NPages - sizeof(chunk meta)
			NOTE: allocated size only include allocated space including the chunk meta )
		
	Space MetaData type (placed before each allocated space):
			prevSize: 4 bytes: size of prev space (0 if first space), does not include spaceMeta
			currentSize: 31 bits: size of this space, does not include spaceMeta
			is_last: 1 bit: is this last space on the chunk?
			chunkOffset: offset (in bytes) to the starting of chunkMeta

	NOTE: 
 */

typedef struct {
	uint16_t allocatedSize;
	uint16_t availableSize;
	uint16_t lastSpaceOffset;
	uint16_t padding;
} ChunkMeta_t;

typedef struct {
	uint16_t prevSize;
	uint16_t currSize:15, isLast:1;
	uint16_t chunkOffset;
	uint16_t padding;
} SpaceMeta_t;


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

/* RESET FREE ALL MEMORY */
__host__ void KernelMallocSimple::reset(){
	RESETBM();
	// reset d_current MallocPage for each thread
	gpuErrchk( cudaMemset(d_currentMallocPage, -1, m_maxNThreads*sizeof(int)) );
}

/* MALLOC FUNCTION */
// helper for setting up a chunk of nPage consecutive pages. Return pageID of the first page
__device__  __inline__ int initChunk(int nPage){
	int pageID = GETXPAGE(nPage);	// consecutive page allocation, ID of first page in the chunk
	void *page = pageAddress(pageID);
	// set chunk metadata
	ChunkMeta_t *chunkMeta = (ChunkMeta_t*)page;
	chunkMeta->availableSize = nPage*PAGE_SIZE - sizeof(ChunkMeta_t);
	chunkMeta->allocatedSize = 0;
	return pageID;
}

__device__ void* KernelMallocSimple::malloc(int size){
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	// retrieve the page we are currently using for malloc
	int pageID = d_currentMallocPage[tid];
	void *page;

	bool need_new_chunk = false;
	// allocate new page if this has not been initialized or current chunk ran out, need to allocate a new chunk
	if (pageID==-1) need_new_chunk = true;
	else {
		// go to actual chunk
		page = pageAddress(pageID);
		ChunkMeta_t *chunkMeta = (ChunkMeta_t*)page;
		if (chunkMeta->availableSize-sizeof(ChunkMeta_t)-chunkMeta->allocatedSize < size+sizeof(SpaceMeta_t))
			need_new_chunk = true;
	}

	if (need_new_chunk){
		// calculate number of pages needed in this chunk
		int nPage = ceil((float)(size+sizeof(ChunkMeta_t))/PAGE_SIZE);
		// set up a new chunk, get the pageID of the first page in the chunk
		pageID = initChunk(nPage);
		// save pageID
		d_currentMallocPage[tid] = pageID;
		page = pageAddress(pageID);
	}

	// once we get here, we should have a chunk ptr that has enough space
	// ptr to chunk is page, chunkMeta is meta of chunk
	// now increment the allocated space on chunk
	ChunkMeta_t *chunkMeta = (ChunkMeta_t*)page;
	SpaceMeta_t *spaceMeta = (SpaceMeta_t*)( (char*)page + sizeof(ChunkMeta_t) +  chunkMeta->allocatedSize);
	SpaceMeta_t *lastSpaceMeta = nullptr;
	void * outPtr = (void*)( (char*)page + sizeof(ChunkMeta_t) + chunkMeta->allocatedSize );

	// update last space meta 's isLast
	if (chunkMeta->allocatedSize)
		lastSpaceMeta = (SpaceMeta_t*) ( (char*)page + chunkMeta->lastSpaceOffset );
	if (lastSpaceMeta){
		spaceMeta->prevSize = lastSpaceMeta->currSize;
		lastSpaceMeta->isLast = false;
	}
	// update current space meta
	spaceMeta->chunkOffset = sizeof(ChunkMeta_t) + chunkMeta->allocatedSize;
	spaceMeta->currSize = size;
	// spaceMeta->prevSize already set above
	spaceMeta->isLast = true;
	// update chunk meta's offset to last space and allocated size 
	chunkMeta->lastSpaceOffset = chunkMeta->allocatedSize;
	chunkMeta->allocatedSize += sizeof(SpaceMeta_t)+size;
	
	return (void*)( (char*)page + spaceMeta->chunkOffset );
}
// end of malloc function


/* FREE FUNCTION */
__device__ void KernelMallocSimple::free(void *addr){

}
// end of free function

#undef CHUNK_METADATA_SIZE





/*
 COLLAB MALLOC:
	Main idea: requests more than 256 gets pointer to start of consecutive pages 
	requests less than 256 add up together and get consecutive pages, than split spaces on that consecutive pages
	Description:
		chunk: a group of consecutive pages
		ChunkMeta: 8bytes at the start of each chunk
		space: a malloc space given to a request
		SpaceMeta: 8bytes before the start of each space

		ChunkMeta:
			- int: pageID
			- int16 occupiedSize: number of bytes still in use in this chunk, when get to 0, free all pages
			- int15 nPages: number of pages in this chunk
			- 1 bit padding

		SpaceMeta:
			- int rootOffset: offset in bytes to get to chunkMeta from start of spaceMeta, 0 if this is a chunkMeta for a single space
			- int31 size: number of bytes in this space (including 8bytes in spaceMeta)
			- 1 bit isChunk: whether this should be interpreted as a chunk, if so, freeing this space would free the chunk
 */
typedef struct {
	uint16_t nPages;
	uint16_t occupiedSize;
	int pageID: 31, padding: 1;
} ChunkMetaCollab_t;

typedef struct {
	unsigned rootOffset;
	unsigned size: 31, isChunk: 1;
} SpaceMetaCollab_t;

/* CONSTRUCTOR */
__host__ KernelMallocCollab::KernelMallocCollab() : MemoryManager()
{
	// no need to do anything here	
}
// end of constructor for KernelMallocSimple

/* DESTRUCTOR */
__host__ KernelMallocCollab::~KernelMallocCollab()
{
	// TODO: free buffer manager
}

/* RESET FREE ALL MEMORY */
__host__ void KernelMallocCollab::reset(){
	RESETBM();
}

/* MALLOC FUNCTION 
	if request size > 256, get pages
	if request size < 256, add them up in a shared mem variable.
	in collaboration, each space has its own spaceMeta
*/
__device__ void* KernelMallocCollab::malloc(int size){
	// round size to next multiple of 8
	size = (size+7)&(-8);
	size += sizeof(SpaceMetaCollab_t);
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	int collab = (size<PAGE_SIZE-sizeof(ChunkMetaCollab_t))? 1 : 0;		// 0 for no collaboration, 1 for collaboration
	int collabMask;		// mask of threads in collaboration
	int collabSuffixSum = 0;	// Suffix-sum of size among threads in collaboration
	int laneID = threadIdx.x%32 + 1;
	__syncwarp();
	if (collab){
		// find all threads who need collaboration
		collabMask = __activemask();
		// calculate Suffix sum with shufle instructions
		int tmpMask = collabMask;
		while (tmpMask){
			int targetLaneID = __ffs(tmpMask);
			tmpMask &= ~(1<<(targetLaneID-1));	// pop the bit at targetLaneID
			int targetSize = __shfl_sync(collabMask, size, targetLaneID);
			if (laneID<=targetLaneID) collabSuffixSum+=targetSize;
		}
		// at this point, the smallest laneID in collabMask, __ffs(collabMask) has to total size
		// offset on the collaborated space would be collabSuffixSum-size
	}

	int nPages; // number of pages this thread needs
	if (!collab)
		nPages = ceilf((float)size/PAGE_SIZE);
	else if (laneID==__ffs(collabMask))
		nPages = ceilf((float)(collabSuffixSum+sizeof(ChunkMetaCollab_t))/PAGE_SIZE);
	else
		nPages = 0;

	// get pages
	int pageID = -1; void *page; ChunkMetaCollab_t *chunkMeta;
	if (nPages){
		pageID = GETXPAGE(nPages);	// consecutive page allocation, ID of first page in the chunk
		page = pageAddress(pageID);
		// set chunk metadata
		chunkMeta = (ChunkMetaCollab_t*)page;
		chunkMeta->nPages = nPages;
		chunkMeta->pageID = pageID;
		chunkMeta->occupiedSize = collab? collabSuffixSum : size;
	}

	// among collab threads, share the page address from its leading laneID
	if (collab){
		pageID = __shfl_sync(collabMask, pageID, __ffs(collabMask)-1);
		page = pageAddress(pageID);
	}

	// calculate space address and set spaceMeta
	SpaceMetaCollab_t *spaceMeta;
	if (!collab){
		spaceMeta = (SpaceMetaCollab_t*)chunkMeta;
		spaceMeta->isChunk = 1;
		return (void*)((char*)page + sizeof(ChunkMetaCollab_t)) ;
	} else {
		int rootOffset = sizeof(ChunkMetaCollab_t) + (collabSuffixSum-size);
		spaceMeta = (SpaceMetaCollab_t*)( (char*)page + rootOffset );
		spaceMeta->isChunk = 0;
		spaceMeta->size = size;
		spaceMeta->rootOffset = rootOffset;
// printf("%d, %u, %d, %d\n", threadIdx.x, spaceMeta->rootOffset, collabSuffixSum, size);
		return (void*)((char*)spaceMeta + sizeof(SpaceMetaCollab_t));
	}
}
// end of malloc function


/* FREE FUNCTION 
	1. go back 8 bytes to check the space meta
	2. if this is a chunkMeta, free all pages in it
	3. if not, go to the chunkMeta:
		- decrease occupiedSize by space size
		- if occupiedSize is 0, free all pages in chunk
*/
__device__ void KernelMallocCollab::free(void *addr){
	SpaceMetaCollab_t *spaceMeta = (SpaceMetaCollab_t*)( (char*)addr - sizeof(SpaceMetaCollab_t) );
	ChunkMetaCollab_t *chunkMeta;
	bool freePages = false;
	if (spaceMeta->isChunk){
		chunkMeta = (ChunkMetaCollab_t*)spaceMeta;
		freePages = true;
	}
	else{
		chunkMeta = (ChunkMetaCollab_t*)( (char*)spaceMeta - spaceMeta->rootOffset );
		// atomicSub((unsigned*)&(chunkMeta->nPages), spaceMeta->size); // need to perform atomic on 4 bytes, starting from nPages
		// if (chunkMeta->occupiedSize<=sizeof(SpaceMetaCollab_t)) freePages = true;
	}
	if (freePages){
		int pageID = chunkMeta->pageID;
		int nPages = chunkMeta->nPages;
		for (int i=0; i<nPages; i++){
			FREEPAGE(pageID+i);
		}
	}
}
// end of free function

#undef CHUNK_METADATA_SIZE
