/* 
	perform one-page allocation, collect TAS, WAS, time
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
#define GRID_FREEPERC_LEN 98
#define N_SAMPLES 96		// number of samples to measure metrics
int GRID_NTHREADS[GRID_NTHREADS_LEN] = {5000};
float GRID_FREEPERC[GRID_FREEPERC_LEN] = {0.1, 0.099, 0.098, 0.097, 0.096, 0.095, 0.094, 0.093, 0.092, 0.091, 0.09, 0.089, 0.088, 0.087, 0.086, 0.085, 0.084, 0.083, 0.082, 0.081, 0.08, 0.079, 0.078, 0.077, 0.076, 0.075, 0.074, 0.073, 0.072, 0.071, 0.07, 0.069, 0.068, 0.067, 0.066, 0.065, 0.064, 0.063, 0.062, 0.061, 0.06, 0.059, 0.058, 0.057, 0.056, 0.055, 0.054, 0.053, 0.052, 0.051, 0.05, 0.049, 0.048, 0.047, 0.046, 0.045, 0.044, 0.043, 0.042, 0.041, 0.04, 0.039, 0.038, 0.037, 0.036, 0.035, 0.034, 0.033, 0.032, 0.031, 0.03, 0.029, 0.028, 0.027, 0.026, 0.025, 0.024, 0.023, 0.022, 0.021, 0.02, 0.019, 0.018, 0.017, 0.016, 0.015, 0.014, 0.013, 0.012, 0.011, 0.01, 0.009, 0.008, 0.007, 0.006, 0.005};


/* Kernel to get 1 page with Random Walk, record step counts */
__global__ void get1page_kernel(int Nthreads, int *d_step_counts){
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if (tid<Nthreads){
		int step_counts;
		int *step_counts_ptr = d_step_counts ? &step_counts : 0;
		auto pageID = getPage(step_counts_ptr);
		if (d_step_counts) d_step_counts[tid] = step_counts;
	}
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