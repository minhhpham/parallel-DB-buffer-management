/* 
	perform one-page allocation on only thread 0 of each warp, collect TAS, WAS, time
	on a grid of Number of threads (N), percentage of available pages (A/T)
	change source file on the first include file to change strategy. No other step is needed
 */

#include "../source/CollabRW_BM.cuh"
#include "metrics.h"
#include <iostream>
#include <iostream>
using namespace std;

// define grid
#define GRID_NTHREADS_LEN 1
#define GRID_FREEPERC_LEN 100
#define N_SAMPLES 100		// number of samples to measure metrics
int GRID_NTHREADS[GRID_NTHREADS_LEN] = {5000};
float GRID_FREEPERC[GRID_FREEPERC_LEN] = {0.5, 0.495, 0.49, 0.485, 0.48, 0.475, 0.47, 0.465, 0.46, 0.455, 0.45, 0.445, 0.44, 0.435, 0.43, 0.425, 0.42, 0.415, 0.41, 0.405, 0.4, 0.395, 0.39, 0.385, 0.38, 0.375, 0.37, 0.365, 0.36, 0.355, 0.35, 0.345, 0.34, 0.335, 0.33, 0.325, 0.32, 0.315, 0.31, 0.305, 0.3, 0.295, 0.29, 0.285, 0.28, 0.275, 0.27, 0.265, 0.26, 0.255, 0.25, 0.245, 0.24, 0.235, 0.23, 0.225, 0.22, 0.215, 0.21, 0.205, 0.2, 0.195, 0.19, 0.185, 0.18, 0.175, 0.17, 0.165, 0.16, 0.155, 0.15, 0.145, 0.14, 0.135, 0.13, 0.125, 0.12, 0.115, 0.11, 0.105, 0.1, 0.095, 0.09, 0.085, 0.08, 0.075, 0.07, 0.065, 0.06, 0.055, 0.05, 0.045, 0.04, 0.035, 0.03, 0.025, 0.02, 0.015, 0.01, 0.005};

/* Kernel to get 1 page with Random Walk, record step counts 
  only lane 0 of a warp gets a page
 */
__global__ void get1page_kernel(int Nthreads, int *d_step_counts){
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
    int step_counts;
	if (tid<Nthreads && (threadIdx.x%32==0)){
		int *step_counts_ptr = d_step_counts ? &step_counts : nullptr;
		auto pageID = getPage(step_counts_ptr);
	}
    // copy step_counts from lane 0 to all lanes so that we have the correct stats later.
    __syncwarp();
    step_counts = __shfl_sync(__activemask(), step_counts, 0);
    if (d_step_counts) d_step_counts[tid] = step_counts;
}

/* make page requests until memory manager has exactly freePercentage */
void fillMemory(float freePercentage){
	int nRequests = (1-freePercentage)*TOTAL_N_PAGES_DEFAULT;
	if (nRequests==0) return;
	std::cerr << "number of pages requested to fill buffer: " << nRequests << std::endl;
	// std::cerr << "filling buffer ...  " << std::endl;
	std::cerr.flush();
	get1page_kernel <<< ceil((float)nRequests/1024), 1024 >>> (nRequests, 0);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
}


/* Execute one kernel of N threads, each gets 1 page with Random Walk
	input: Nthreads: 		number of threads
	return: *avgStep:		average of step counts across all threads
			*avgMaxWarp:	average of Max of Warp across all warps
			*runTime:		total run time (s)
 */
Metrics_t measureMetrics(int Nthreads, float freePercentage){
	// run kernel until get to desired free percentage
	prefillBuffer(freePercentage);

	// allocate metrics array on host
	int *h_step_counts = (int*)malloc((1<<20)*sizeof(int));
	// allocate metrics array on gpu
	int *d_step_counts;
	gpuErrchk( cudaMalloc((void**)&d_step_counts, (1<<20)*sizeof(int)) );

	// execute kernel;
	KernelTiming timing;
	timing.startKernelTiming();
	get1page_kernel <<< ceil((float)Nthreads/32), 32 >>> (Nthreads, d_step_counts);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
	float total_time = timing.stopKernelTiming();

	// copy metrics to host
	gpuErrchk( cudaMemcpy(h_step_counts, d_step_counts, Nthreads*sizeof(int), cudaMemcpyDeviceToHost) );

	// aggregate metrics and return
	Metrics_t out = aggregate_metrics(h_step_counts, Nthreads);
	out.runTime = total_time;

	free(h_step_counts); cudaFree(d_step_counts);
	return out;
}


int main(int argc, char const *argv[])
{
	// gpuErrchk(cudaSetDevice(0));

	/* initialize system */
	fprintf(stderr, "initializing memory management system ... \n");
	initMemoryManagement(2, 2147);

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