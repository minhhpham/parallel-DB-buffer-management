/* 
	perform one-page allocation, save ptr on global mem, then free
	measure and print time of free kernel
	on a grid of Number of threads (N), percentage of available pages (A/T)
	change source file on the first include file to change strategy. No other step is needed
 */

#include "../source/RandomWalkBasic.cuh"
#include "metrics.h"
#include <iostream>

// define grid
#define GRID_NTHREADS_LEN 21
#define GRID_FREEPERC_LEN 1
#define N_SAMPLES 30		// number of samples to measure metrics
int GRID_NTHREADS[GRID_NTHREADS_LEN] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576};
float GRID_FREEPERC[GRID_FREEPERC_LEN] = {1};


/* Kernel to get 1 page with Random Walk, record step counts */
__global__ void get1page_kernel(int Nthreads, int *d_step_counts){
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if (tid<Nthreads){
		int step_counts;
		int *tmp = d_step_counts? &step_counts : 0;
		int pageID = getPage(tmp);
		if (d_step_counts) d_step_counts[tid] = step_counts;
	}
}


__global__ void getPage_kernel(int Nthreads, int *d_allocated_pageID){
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if (tid<Nthreads)
		d_allocated_pageID[tid] = getPage();
}

__global__ void freePage_kernel(int Nthreads, int *d_allocated_pageID){
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if (tid<Nthreads){
		int pageID = d_allocated_pageID[tid];
		freePage(pageID);
	}
}


/* make page requests until memory manager has exactly freePercentage */
void fillMemory(float freePercentage){
	int nRequests = (1-freePercentage)*TOTAL_N_PAGES_DEFAULT;
	if (nRequests==0) return;
	// std::cerr << "number of pages requested to fill buffer: " << nRequests << std::endl;
	// std::cerr << "filling buffer ...  " << std::endl;
	std::cerr.flush();
	get1page_kernel <<< ceil((float)nRequests/1024), 1024 >>> (nRequests, 0);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
	// std::cerr << "free page percentage: " << 100*getFreePagePercentage() << std::endl;
}


/* Execute one kernel of N threads, each gets 1 page with Random Walk
	input: Nthreads: 		number of threads
	return: *avgStep:		average of step counts across all threads
			*avgMaxWarp:	average of Max of Warp across all warps
			*runTime:		total run time (s)
 */
Metrics_t measureMetrics(int Nthreads, float freePercentage){
	// run kernel until get to desired free percentage
	fillMemory(freePercentage);

	int *d_allocated_pageID;
	gpuErrchk( cudaMalloc((void**)&d_allocated_pageID, (1<<20)*sizeof(int)) );

	// execute get page kernel
	getPage_kernel <<< ceil((float)Nthreads/32), 32 >>> (Nthreads, d_allocated_pageID);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );

	// execute free page kernel and  measure time
	KernelTiming timing;
	timing.startKernelTiming();
	freePage_kernel <<< ceil((float)Nthreads/32), 32 >>> (Nthreads, d_allocated_pageID);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
	float total_time = timing.stopKernelTiming();

	// aggregate metrics and return
	Metrics_t out;
	out.runTime = total_time;

	cudaFree(d_allocated_pageID);
	return out;
}


int main(int argc, char const *argv[])
{
	// gpuErrchk(cudaSetDevice(0));

	/* initialize system */
	fprintf(stderr, "initializing memory management system ... \n");
	initMemoryManagement();

	/* run getpage on a grid (NThreads, freePercentage) and get metrics */
	std::cout << "T,N,A%%,Average_steps,Average_Max_Warp,Time(ms)" << std::endl;
	for (int i=0; i<GRID_FREEPERC_LEN; i++){
		for (int j=0; j<GRID_NTHREADS_LEN; j++){
			float freePercentage = GRID_FREEPERC[i];
			int Nthreads = GRID_NTHREADS[j];
			Metrics_t *metrics_array = (Metrics_t*)malloc(N_SAMPLES*sizeof(Metrics_t));	// array of metrics samples
			for (int k=0; k<N_SAMPLES; k++){
				resetMemoryManager();
				metrics_array[k] = measureMetrics(Nthreads, freePercentage);
			}
			Metrics_t average_metrics = sample_average(metrics_array, N_SAMPLES);
			// print results
			std::cout << TOTAL_N_PAGES_DEFAULT << "," << Nthreads << "," << freePercentage << "," << average_metrics.avgStep << "," << average_metrics.avgMaxWarp << "," << average_metrics.runTime << std::endl;
			std::cout.flush();
		}
	}
	return 0;
}