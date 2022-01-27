/* BECHMARK 1: Allocation Performance for Varying Allocation Size
	Test 10,000 and 100,000 allocations in the range between 4B-X where X is from 4B to 8192B
*/

#include "parallelPage.cuh"
#include "malloc.cuh"

/* RNG function to replace curand, better efficiency */
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


/* kernel: each thread allocate size */
__global__ void allocate_kernel(int maxSize, int NThreads, KernelMallocSimple memManager){
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if (tid<NThreads){
		int seed = (tid<<15) + (int16_t)clock();
		seed = RNG_LCG(seed);
		int size = (unsigned)seed % (maxSize-3);
		size += 4;
		void* ptr = memManager.malloc(size);
	}
}

float testAllocate(int NThreads, int size, KernelMallocSimple memManager){
	// warm-up 
	allocate_kernel <<< ceil((float)NThreads/32), 32 >>> (4, NThreads, memManager);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );

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
	printf("Simple Malloc based on Random Walk Buffer\n");
	printf("Allocation_size,10K Time (ms),100K Time (ms)\n");

	const int N_SAMPLES = 30;
	for (int max_allocation_size=4; max_allocation_size<=8192; max_allocation_size+=4){
		// 10K threads
		// run experiment N_SAMPLES times
		float times[N_SAMPLES];
		for (int i=0; i<N_SAMPLES; i++){
			memManager.reset();
			times[i] = testAllocate(100000, max_allocation_size, memManager);
		}
		// take average across samples
		float sum = 0.0;
		for (int i=0; i<N_SAMPLES; i++)
			sum += times[i];
		float avg10K = sum/N_SAMPLES;

		// 100K threads
		// run experiment N_SAMPLES times
		for (int i=0; i<N_SAMPLES; i++){
			memManager.reset();
			times[i] = testAllocate(1000000, max_allocation_size, memManager);
		}
		// take average across samples
		sum = 0.0;
		for (int i=0; i<N_SAMPLES; i++)
			sum += times[i];
		float avg100K = sum/N_SAMPLES;


		// print result
		printf("%d,%f,%f\n", max_allocation_size, avg10K, avg100K);
	}

	return 0;
}