#include "parallelPage.cuh"
#include "malloc.cuh"

/* kernel: each thread allocate size */
__global__ void allocate_kernel(int size, int NThreads, KernelMallocSimple memManager){
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if (tid<NThreads){
		void* ptr = memManager.malloc(size);
		// int pageID = getPageRandomWalk_BM();

	}
}

float testAllocate(int NThreads, int size, KernelMallocSimple memManager){
	// start timer 
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	allocate_kernel <<< ceil((float)NThreads/32), 32 >>> (size, NThreads, memManager);
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

int main(int argc, char const *argv[])
{
	KernelMallocSimple memManager(1000000);
	float time = testAllocate(5000, 4, memManager);
	printf("%f\n", time);

	return 0;
}