/* 
	perform one-page allocation, collect TAS, WAS, time
	on a grid of Number of threads (N), percentage of available pages (A/T)
	change source file on the first include file to change strategy. No other step is needed
 */

#include "../source/CollabRW_BM.cuh"
#include "metrics.h"
#include <iostream>

// define grid
#define GRID_NTHREADS_LEN 10
#define GRID_FREEPERC_LEN 6
#define N_SAMPLES 1		// number of samples to measure metrics
int GRID_NTHREADS[GRID_NTHREADS_LEN] = {1000, 2000, 3000, 4000, 5000, 10000, 20000, 30000, 50000, 100000};
float GRID_FREEPERC[GRID_FREEPERC_LEN] = {1.0, 0.5, 0.2, 0.1, 0.05, 0.01};


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

	// allocate metrics array on host
	int *h_step_counts = (int*)malloc(100000*sizeof(int));
	// allocate metrics array on gpu
	int *d_step_counts;
	gpuErrchk( cudaMalloc((void**)&d_step_counts, 100000*sizeof(int)) );

	// execute kernel;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	get1page_kernel <<< ceil((float)Nthreads/32), 32 >>> (Nthreads, d_step_counts);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float total_time;
	cudaEventElapsedTime(&total_time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

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
	initMemoryManagement();

	/* run getpage on a grid (NThreads, freePercentage) and get metrics */
	std::cout << "T,N,A%%,Average_steps,Average_Max_Warp,Time(ms)" << std::endl;
	for (int i=0; i<GRID_NTHREADS_LEN; i++){
		for (int j=0; j<GRID_FREEPERC_LEN; j++){
			int Nthreads = GRID_NTHREADS[i];
			float freePercentage = GRID_FREEPERC[j];
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