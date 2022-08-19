/* 
	perform arbitrary size allocation (up to 16K)
	on a grid of Number of threads (N), percentage of available pages (A/T)
	change source file on the first include file to change strategy. No other step is needed
 */

#include "../source/RWMalloc.cuh"
#include "metrics.h"
#include <iostream>

// define grid
#define GRID_NTHREADS_LEN 2
#define GRID_FREEPERC_LEN 3
#define GRID_ALLOCSIZE_LEN 37
#define N_SAMPLES 100		// number of samples to measure metrics
int GRID_NTHREADS[GRID_NTHREADS_LEN] = {10000, 100000};
float GRID_FREEPERC[GRID_FREEPERC_LEN] = {1};
int GRID_ALLOCSIZE[GRID_ALLOCSIZE_LEN] = {4,16,64,128,192,256,512,768,1024,1280,1536,1792,2048,2304,2560,2816,3072,3328,3584,3840,4096,4352,4608,4864,5120,5376,5632,5888,6144,6400,6656,6912,7168,7424,7680,7936,8192};


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

/* Kernel to get n pages with Random Walk, record step counts */
__global__ void malloc_kernel(int Nthreads, int *d_step_counts, int size){
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if (tid<Nthreads)
		void * ptr = mallocRW(size);
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
	// std::cerr.flush();
}


/* Execute one kernel of N threads, each gets n pages
	input: Nthreads: 		number of threads
	input: n:				number of consecutive pages, 0 for random number of pages
	return: *avgStep:		average of step counts across all threads
			*avgMaxWarp:	average of Max of Warp across all warps
			*runTime:		total run time (s)
 */
Metrics_t measureMetrics(int Nthreads, float freePercentage, int size){
	// run kernel until get to desired free percentage
	fillMemory(freePercentage);

	// allocate metrics array on host
	int *h_step_counts = (int*)malloc((1<<20)*sizeof(int));
	// allocate metrics array on gpu
	int *d_step_counts;
	gpuErrchk( cudaMalloc((void**)&d_step_counts, (1<<20)*sizeof(int)) );

	// execute kernel;
	KernelTiming timing;
	timing.startKernelTiming();
	malloc_kernel <<< ceil((float)Nthreads/32), 32 >>> (Nthreads, d_step_counts, size);
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
	/* command descriptions */
	if(argc>1 && ((strncmp(argv[1], "-h", 2) == 0) || (strncmp(argv[1], "-help", 4) == 0))){
		fprintf(stderr, "USAGE: ./unitTest2 [options]\n");
		fprintf(stderr, "OPTIONS: -s : allocation size. Default 16 \n");
		return 0;
	}
	/* parse options */
	int allocation_size = 16;
	for (int i=0; i<argc; i++){
		if(strncmp(argv[i], "-s", 2) == 0)
			allocation_size = atoi(argv[i]);
	}


	/* initialize system */
	fprintf(stderr, "initializing memory management system ... \n");
	initMemoryManagement();

	/* run getpage on a grid (NThreads, freePercentage) and get metrics */
	std::cout << "N,A%%,Average_steps,Average_Max_Warp,Time(ms)" << std::endl;
	for (int i=0; i<GRID_NTHREADS_LEN; i++){
		for (int j=0; j<GRID_ALLOCSIZE_LEN; j++){
			float freePercentage = 1;
			int Nthreads = GRID_NTHREADS[i];
			int allocation_size = GRID_ALLOCSIZE[j];
			Metrics_t *metrics_array = (Metrics_t*)malloc(N_SAMPLES*sizeof(Metrics_t));	// array of metrics samples
			for (int k=0; k<N_SAMPLES; k++){
				resetMemoryManager();
				metrics_array[k] = measureMetrics(Nthreads, freePercentage, allocation_size);
			}
			Metrics_t average_metrics = sample_average(metrics_array, N_SAMPLES);
			// print results
			std::cout << Nthreads << "," << allocation_size << "," << average_metrics.avgStep << "," << average_metrics.avgMaxWarp << "," << average_metrics.runTime << std::endl;
			std::cout.flush();
		}
	}
	return 0;
}