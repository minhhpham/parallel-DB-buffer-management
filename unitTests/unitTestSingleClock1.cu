/* collect steps and time for Single clock under different percentages of free page
 */

#include "parallelPage.cuh"
#include "metrics.h"

/* fill up the buffer so that p% of pages are free*/
__global__ void fillBuffer_kernel(float p){
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if (tid>=TOTAL_N_PAGES) return;
	int mod = ceil((float)100/p);
	int pageID = getPageSingleClock();
	if (tid%mod==0)
		freePageSingleClock(pageID);
}
/* fill up the buffer so that p% of pages are free*/
void fillBuffer(float p){
	resetBufferSingleClock();
	fillBuffer_kernel <<< ceil((float)TOTAL_N_PAGES/32), 32 >>> (p);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
}

/* kernel to collect steps
d_stepCounts is array with length Nthreads
 */
__global__ void collectStep_kernel(int Nthreads, int *d_stepCounts){
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if (tid>=Nthreads) return;
	int step_counts;
	int *tmp = d_stepCounts? &step_counts : 0;
	int pageID = getPageSingleClock(tmp);
	if (d_stepCounts) d_stepCounts[tid] = step_counts;
}

Metrics_t collectData(int Nthreads){
	// allocate stepCounts array on gpu
	int *d_stepCounts;
	gpuErrchk( cudaMalloc((void**)&d_stepCounts, Nthreads*sizeof(int)) );

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	// run kernel
	collectStep_kernel <<< ceil((float)Nthreads/32), 32 >>> (Nthreads, d_stepCounts);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float totalTime;
	cudaEventElapsedTime(&totalTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// copy output to host
	int *h_stepCounts = (int*)malloc(Nthreads*sizeof(int));
	gpuErrchk( cudaMemcpy(h_stepCounts, d_stepCounts, Nthreads*sizeof(int), cudaMemcpyDeviceToHost) );

	// aggregate steps and record time
	Metrics_t out = aggregate_metrics(h_stepCounts, Nthreads);
	out.runTime = totalTime;

	// done
	cudaFree(d_stepCounts); free(h_stepCounts);
	return out;
}



int main(int argc, char const *argv[])
{
	initPagesSingleClock();

	/* collect data at 50% free */
	fillBuffer(50);
	printNumPagesLeftSingleClock();
	for (int Nthreads=1; Nthreads<5000; Nthreads+=50){
		fillBuffer(50);
		Metrics_t metrics = collectData(Nthreads);
		// print results to stdout
		printf("%d,%d,%d,%f,%f,%f\n", TOTAL_N_PAGES, Nthreads, (int)(TOTAL_N_PAGES*0.5), metrics.avgStep, metrics.avgMaxWarp, metrics.runTime);
	}

	/* collect data at 1% free */
	fillBuffer(1);
	printNumPagesLeftSingleClock();
	for (int Nthreads=1; Nthreads<5000; Nthreads+=50){
		fillBuffer(1);
		Metrics_t metrics = collectData(Nthreads);
		// print results to stdout
		printf("%d,%d,%d,%f,%f,%f\n", TOTAL_N_PAGES, Nthreads, (int)(TOTAL_N_PAGES*0.01), metrics.avgStep, metrics.avgMaxWarp, metrics.runTime);
	}

	/* collect data at 0.7% free */
	fillBuffer(0.7);
	printNumPagesLeftSingleClock();
	for (int Nthreads=1; Nthreads<5000; Nthreads+=50){
		fillBuffer(0.7);
		Metrics_t metrics = collectData(Nthreads);
		// print results to stdout
		printf("%d,%d,%d,%f,%f,%f\n", TOTAL_N_PAGES, Nthreads, (int)(TOTAL_N_PAGES*0.007), metrics.avgStep, metrics.avgMaxWarp, metrics.runTime);
	}

	return 0;
}