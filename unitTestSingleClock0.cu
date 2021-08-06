/* 
	this script print out clock cycles of Single Clock as number of threads increase
	Assuming that all threads find a free page on first try at the current arm
 */

#include "parallelPage.cuh"
#include "metrics.h"
#define MAX_THREADS 50000

/* in this kernel, each thread gets and frees 30 pages and record the average number of cycles taken to d_nCycles 
*/
#define N_SAMPLES 30
__global__ void getPageFreePage_kernel(int *d_getCycles, int *d_freeCycles){
	int getCycles[N_SAMPLES];
	int freeCycles[N_SAMPLES];
	for (int i=0; i<N_SAMPLES; i++){
		// get page 
		volatile unsigned long long start = (unsigned long long)clock();
		volatile int pageID = getPageSingleClock();
		volatile unsigned long long stop = (unsigned long long)clock();
		getCycles[i] = (int)(stop - start);
		// free page
		start = (unsigned long long)clock();
		freePageSingleClock(pageID);
		stop = (unsigned long long)clock();
		freeCycles[i] = (int)(stop - start);
	}
	// take average across N_SAMPLES
	unsigned long long getCyclesSum = 0, freeCyclesSum = 0;
	for (int i=0; i<N_SAMPLES; i++){
		getCyclesSum = getCyclesSum + getCycles[i];
		freeCyclesSum = freeCyclesSum + freeCycles[i];
	}
	// record cycles numbers to output
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	d_getCycles[tid] = (int)(getCyclesSum/N_SAMPLES);
	d_freeCycles[tid] = (int)(freeCyclesSum/N_SAMPLES);
}

typedef struct
{
	int getPage;
	int freePage;
} cycles_t;

/*this function return the average number of cycles to getPage and freePage*/
cycles_t testCyclesGetPage(int Nthreads){
	// allocate memory on device for cycle recording
	int *d_getCycles, *d_freeCycles;
	gpuErrchk( cudaMalloc(&d_getCycles, Nthreads*sizeof(int)) );
	gpuErrchk( cudaMalloc(&d_freeCycles, Nthreads*sizeof(int)) );
	// launch kernel
	getPageFreePage_kernel <<< ceil((float)Nthreads/32), 32 >>> (d_getCycles, d_freeCycles);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );

	// copy cycles numbers to host
	int *h_getCycles = (int*)malloc(Nthreads*sizeof(int));
	int *h_freeCycles = (int*)malloc(Nthreads*sizeof(int));
	gpuErrchk( cudaMemcpy(h_getCycles, d_getCycles, Nthreads*sizeof(int), cudaMemcpyDeviceToHost) );
	gpuErrchk( cudaMemcpy(h_freeCycles, d_freeCycles, Nthreads*sizeof(int), cudaMemcpyDeviceToHost) );
	cudaFree(d_getCycles); cudaFree(d_freeCycles);

	// take average
	cycles_t cycles;
	unsigned long long sumGet = 0, sumFree = 0;
	for (int i=0; i<Nthreads; i++){
		sumGet += h_getCycles[i];
		sumFree += h_freeCycles[i];
	}
	cycles.getPage = (int)(sumGet)/Nthreads;
	cycles.freePage = (int)(sumFree)/Nthreads;

	free(h_getCycles); free(h_freeCycles);
	return cycles;
}

__global__ void fillUpBuffer(){
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	int pageID = getPageSingleClock();
	if (tid%2)
		freePageSingleClock(pageID);
}

int main(int argc, char const *argv[])
{
	/* init Single Clock Buffer */
	initPagesSingleClock();
	// fillUpBuffer <<< ceil((float)TOTAL_N_PAGES/32), 32 >>> ();
	// gpuErrchk( cudaPeekAtLastError() );
	// gpuErrchk( cudaDeviceSynchronize() );

	/* run test for 1 thread*/
	printf("Nthreads, getPage Cycles, freePageCycles\n");
	cycles_t cycles = testCyclesGetPage(1);
	printf("1,%d,%d\n", cycles.getPage, cycles.freePage);

	/* run test for multiple threads */
	for (int nThreads=32; nThreads<MAX_THREADS; nThreads+=32){
		cycles = testCyclesGetPage(nThreads);
		printf("%d,%d,%d\n", nThreads, cycles.getPage, cycles.freePage);
	}

	return 0;
}