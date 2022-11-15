/*
	perform X number of allocations per warp, defined by GRID_NWARPALLOCATION
	on a grid of Number of threads (N), percentage of available pages (A/T)
	change source file on the first include file to change strategy. No other step is needed
	only measured time. step count not yet implemented
 */

#include "../source/CollabRW_BM.cuh"
#include "metrics.h"
#include <iostream>
using namespace std;

// define grid
#define GRID_NTHREADS_LEN 1
#define GRID_FREEPERC_LEN 5
#define N_SAMPLES 10000 // number of samples to measure metrics
#define N_WARPALLOCATION 16
int GRID_NTHREADS[GRID_NTHREADS_LEN] = {5000};
float GRID_FREEPERC[GRID_FREEPERC_LEN] = {0.05, 0.03, 0.01, 0.008, 0.005};
int GRID_NWARPALLOCATION[N_WARPALLOCATION] = {32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17};

/* Kernel to get 1 page with Random Walk, record step counts
  Nalloc: number of allocations per warp
  only lane 0 of a warp gets a page
 */
__global__ void get1page_kernel(int Nthreads, int Nalloc, int *d_step_counts)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int step_counts;
	if (tid < Nthreads && ((threadIdx.x % 32) < Nalloc))
	{
		int *step_counts_ptr = d_step_counts ? &step_counts : nullptr;
		auto pageID = getPage(step_counts_ptr);
		if (d_step_counts)
			d_step_counts[tid] = step_counts;
	}
}

/* Execute one kernel of N threads, each gets 1 page with Random Walk
	input: Nthreads: 		number of threads
	return: *avgStep:		average of step counts across all threads
			*avgMaxWarp:	average of Max of Warp across all warps
			*runTime:		total run time (s)
 */
Metrics_t measureMetrics(int Nthreads, float freePercentage, int Nalloc)
{
	// run kernel until get to desired free percentage
	prefillBuffer(freePercentage);

	// // allocate metrics array on host
	// int *h_step_counts = (int*)malloc((1<<20)*sizeof(int));
	// // allocate metrics array on gpu
	// int *d_step_counts;
	// gpuErrchk( cudaMalloc((void**)&d_step_counts, (1<<20)*sizeof(int)) );

	// execute kernel;
	KernelTiming timing;
	timing.startKernelTiming();
	get1page_kernel<<<ceil((float)Nthreads / 32), 32>>>(Nthreads, Nalloc, nullptr);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	float total_time = timing.stopKernelTiming();

	// copy metrics to host
	// gpuErrchk( cudaMemcpy(h_step_counts, d_step_counts, Nthreads*sizeof(int), cudaMemcpyDeviceToHost) );

	// aggregate metrics and return
	Metrics_t out;
	// Metrics_t out = aggregate_metrics(h_step_counts, Nthreads);
	out.runTime = total_time;

	// free(h_step_counts); cudaFree(d_step_counts);
	return out;
}

int main(int argc, char const *argv[])
{
	// gpuErrchk(cudaSetDevice(0));

	/* initialize system */
	fprintf(stderr, "initializing memory management system ... \n");
	initMemoryManagement(2, 2147);

	/* run getpage on a grid (NThreads, freePercentage) and get metrics */
	std::cout << "T,N,A%%,Nalloc,Average_steps,Average_Max_Warp,Time(ms)" << std::endl;
	for (float freePercentage : GRID_FREEPERC)
	{
		for (int Nthreads : GRID_NTHREADS)
		{
			for (int Nalloc : GRID_NWARPALLOCATION)
			{
				Metrics_t *metrics_array = (Metrics_t *)malloc(N_SAMPLES * sizeof(Metrics_t)); // array of metrics samples
				for (int k = 0; k < N_SAMPLES; k++)
				{
					resetMemoryManager();
					metrics_array[k] = measureMetrics(Nthreads, freePercentage, Nalloc);
				}
				Metrics_t average_metrics = sample_average(metrics_array, N_SAMPLES);
				// print results
				std::cout << h_total_n_pages<< "," << Nthreads << "," << freePercentage << "," << Nalloc << "," << average_metrics.avgStep << "," << average_metrics.avgMaxWarp << "," << average_metrics.runTime << std::endl;
				std::cout.flush();
			}
		}
	}
	return 0;
}