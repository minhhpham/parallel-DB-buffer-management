#include <bits/stdc++.h>

class MemoryManager
{
public:
	__host__ MemoryManager(int _maxNThreads = INT_MAX);	// constructor, maxNThreads = max number of threads supported
	__host__ ~MemoryManager();							// destructor
	__host__ void reset();								// free all memory
	__device__ void *malloc(int size);					// memory allocation
	__device__ void free(void *addr);					// memory free
	int m_maxNThreads;									// max number of threads supported
};

class KernelMallocSimple : public MemoryManager
{
public:
	__host__ KernelMallocSimple(int _maxNThreads);
	__host__ ~KernelMallocSimple();
	__host__ void reset();						// free all memory
	__device__ void *malloc(int size);			// memory allocation
	__device__ void free(void *addr);			// memory free
private:
	int *d_currentMallocPage;					// device array to keep track of the page each thread is using for malloc
};


class KernelMallocCollab : public MemoryManager
{
public:
	__host__ KernelMallocCollab();
	__host__ ~KernelMallocCollab();
	__host__ void reset();						// free all memory
	__device__ void *malloc(int size);			// memory allocation
	__device__ void free(void *addr);			// memory free
private:
	int *d_currentMallocPage;					// device array to keep track of the page each thread is using for malloc
};
