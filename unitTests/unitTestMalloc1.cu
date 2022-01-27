/* BECHMARK 1: Allocation Performance for Allocation Size
	Test 10,000 and 100,000 allocations in the range between 4B-8192B

*/

#include "parallelPage.cuh"
#include "malloc.cuh"
void ** d_ptrs; // array on GPU to save allocated pointers

#define LCG_M 8188
#define LCG_A 1103515245
#define LCG_C 12345
/* kernel: each thread allocate size */
__global__ void allocateRandom_kernel(int NThreads, KernelMallocCollab memManager, void **d_ptrs){
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned size = (unsigned)((LCG_A*(long long)tid) + LCG_C)%(LCG_M) + 4;
	void* ptr;
	if (tid<NThreads){
		ptr = memManager.malloc(size);
	}
	d_ptrs[tid] = ptr;
}



/* kernel: each thread allocate size */
__global__ void allocate_kernel(int size, int NThreads, KernelMallocCollab memManager, void **d_ptrs){
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	void* ptr;
	if (tid<NThreads){
		ptr = memManager.malloc(size);
	}
	d_ptrs[tid] = ptr;
}

__global__ void free_kernel(void **d_ptrs, int NThreads, KernelMallocCollab memManager){
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if (tid<NThreads){
		memManager.free(d_ptrs[tid]);
	}
}


float testAllocateRandom(int NThreads, KernelMallocCollab memManager, float freePercent){
	// get pages until we reach freePercent
	int nPages = TOTAL_N_PAGES/100*(100-freePercent) ; // number of pages to obtain
	for (int pages=0; pages+(1<<20)<=nPages; pages+=(1<<20)){
		allocate_kernel <<< ceil((float)(1<<20)/32), 32 >>> (PAGE_SIZE-8, (1<<20), memManager, d_ptrs);
	}
	printNumPagesLeftCollabRW_BM();
	////////////////////////// 
	// warm-up 
	allocateRandom_kernel <<< ceil((float)NThreads/32), 32 >>> (NThreads, memManager, d_ptrs);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
	free_kernel <<< ceil((float)NThreads/32), 32 >>> (d_ptrs, NThreads, memManager);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );

	// start timer 
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	allocateRandom_kernel <<< ceil((float)NThreads/32), 32 >>> (NThreads, memManager, d_ptrs);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );

	// stop timer
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsed_time;
	cudaEventElapsedTime(&elapsed_time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	return elapsed_time;
}



float testAllocate(int NThreads, int size, KernelMallocCollab memManager, float freePercent){
	// get pages until we reach freePercent
	int nPages = TOTAL_N_PAGES/100*(100-freePercent) ; // number of pages to obtain
	for (int pages=0; pages+(1<<20)<=nPages; pages+=(1<<20)){
		// printNumPagesLeftCollabRW_BM();
		allocate_kernel <<< ceil((float)(1<<20)/32), 32 >>> (PAGE_SIZE-8, (1<<20), memManager, d_ptrs);
	}
	// printNumPagesLeftCollabRW_BM();
	////////////////////////// 
	// warm-up 
	allocate_kernel <<< ceil((float)NThreads/32), 32 >>> (size, NThreads, memManager, d_ptrs);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
	free_kernel <<< ceil((float)NThreads/32), 32 >>> (d_ptrs, NThreads, memManager);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );

	// start timer 
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	allocate_kernel <<< ceil((float)NThreads/32), 32 >>> (size, NThreads, memManager, d_ptrs);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );

	// stop timer
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsed_time;
	cudaEventElapsedTime(&elapsed_time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	return elapsed_time;
}

float testAllocateScenario1(int NThreads, int size, KernelMallocCollab memManager, float freePercent){
	// get pages until we reach freePercent
	int nPages = TOTAL_N_PAGES/100*(100-freePercent) ; // number of pages to obtain
	for (int pages=0; pages+(1<<20)<=nPages; pages+=(1<<20)){
		allocate_kernel <<< ceil((float)(1<<20)/32), 32 >>> (PAGE_SIZE-8, (1<<20), memManager, d_ptrs);
	}
	////////////////////////// 
	// warm-up 
	allocate_kernel <<< ceil((float)NThreads/32), 32 >>> (size, NThreads, memManager, d_ptrs);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
	free_kernel <<< ceil((float)NThreads/32), 32 >>> (d_ptrs, NThreads, memManager);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );

	// start timer 
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	allocate_kernel <<< ceil((float)NThreads/32), 32 >>> (size, NThreads, memManager, d_ptrs);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );

	// stop timer
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsed_time;
	cudaEventElapsedTime(&elapsed_time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	free_kernel <<< ceil((float)NThreads/32), 32 >>> (d_ptrs, NThreads, memManager);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );


	return elapsed_time;

}



int main(int argc, char const *argv[])
{
	KernelMallocCollab memManager;
	printf("Simple Malloc based on Random Walk Buffer\n");
	printf("Allocation_size,FreePercent,NThreads,Time (ms)\n");

	const int N_SAMPLES = 10;
	const float freePercents[4] = {0.7, 10, 50, 100}; 

	gpuErrchk( cudaMalloc(&d_ptrs, (1<<20)*sizeof(void*)) );

	// change allocation_size below to create scenario. Use testAllocateScenario1 for scenario 1
	int allocation_size = 16;
	for (int i=0; i<4; i++){	// index to freePercents
		float freePercent = freePercents[i];
		for (int NThreads=1000; NThreads<=30000; NThreads+=1000){
			// run experiment N_SAMPLES times
			float times[N_SAMPLES];
			for (int i=0; i<N_SAMPLES; i++){
				memManager.reset();
				times[i] = testAllocateRandom (NThreads, memManager, freePercent);
			}
			// take average across samples
			float sum = 0.0;
			for (int i=0; i<N_SAMPLES; i++)
				sum += times[i];
			float avg = sum/N_SAMPLES;

			// print result
			printf("%d,%f,%d,%f\n", allocation_size, freePercent, NThreads, avg);
		}
		printf("\n");
	}

	return 0;
}