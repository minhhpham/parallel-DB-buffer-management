#include "parallelPage.cuh"
#include "metrics.h"

static __device__ void fillPage(void *page){
	char *ptr = (char*)page;
	for (int i=0; i<PAGE_SIZE; i++)
		ptr[i] = 1;
}

/* sequential get page */
#define NTRIALS 30
#define BURN	5	// burn 5 trials to warm up
// d_pageIdx is an 35*Nthreads array
// same as d_counts
// only 1 thread running this kernel
__global__ void LinkedList_getPage_Sequential_kernel(int Nthreads, int *d_pageIdx, int *d_counts){
	for (int trial=0; trial<(NTRIALS+BURN); trial++){
		for (int tid=0; tid<Nthreads; tid++){
			volatile unsigned long long start = (unsigned long long)clock();
			int pageID = getPageLinkedList();
			volatile unsigned long long stop = (unsigned long long)clock();
			// save pageID for freeing kernel
			d_pageIdx[trial*Nthreads+tid] = pageID;
			// save metrics
			d_counts[trial*Nthreads+tid] = (int)(stop - start);
		}
	}

	// aggregate time (cycles)
	unsigned long long threadAvg = 0;
	for (int tid=0; tid<Nthreads; tid++){
		unsigned long long Avg = 0;
		// average accross trials
		for (int trial=BURN; trial<(NTRIALS+BURN); trial++)
			Avg+=d_counts[trial*Nthreads+tid];
		Avg = Avg/NTRIALS;
		// average accross all threads
		threadAvg = (threadAvg*tid + Avg)/(tid+1);
		// print result
		printf("Nthreads=%d,clocks=%lu\n",tid+1, threadAvg);
	}
}

/* sequential free page */
__global__ void LinkedList_freePage_Sequential_kernel(int Nthreads, int *d_pageIdx, int *d_counts){
	for (int trial=0; trial<(NTRIALS+BURN); trial++){
		for (int tid=0; tid<Nthreads; tid++){
			int pageID = d_pageIdx[trial*Nthreads+tid];
			volatile unsigned long long start = (unsigned long long)clock();
			freePageLinkedList(pageID);
			volatile unsigned long long stop = (unsigned long long)clock();
			// save metrics
			d_counts[trial*Nthreads+tid] = (int)(stop - start);
		}
	}

	// aggregate time (cycles)
	unsigned long long threadAvg = 0;
	for (int tid=0; tid<Nthreads; tid++){
		unsigned long long Avg = 0;
		// average accross trials
		for (int trial=BURN; trial<(NTRIALS+BURN); trial++)
			Avg+=d_counts[trial*Nthreads+tid];
		Avg = Avg/NTRIALS;
		// average accross all threads
		threadAvg = (threadAvg*tid + Avg)/(tid+1);
		// print result
		printf("Nthreads=%d,clocks=%lu\n",tid+1, threadAvg);
	}
}

/* Concurrent get page, same logic as sequential get page */
__global__ void LinkedList_getPage_Concurrent_kernel(int Nthreads, int *d_pageIdx, int *d_counts){
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	for (int trial=0; trial<(NTRIALS+BURN); trial++){
		volatile unsigned long long start = (unsigned long long)clock();
		int pageID = getPageLinkedList();
		volatile unsigned long long stop = (unsigned long long)clock();
		// save pageID for freeing kernel
		d_pageIdx[trial*Nthreads+tid] = pageID;
		// save metrics
		d_counts[trial*Nthreads+tid] = (int)(stop - start);
	}

	// aggregate time (cycles) across trials, save to row 0 of d_counts
	unsigned long long Avg = 0;
	for (int trial=BURN; trial<(NTRIALS+BURN); trial++)
		Avg+=d_counts[trial*Nthreads+tid];
	Avg = Avg/NTRIALS;
	d_counts[tid] = (int)Avg;
}

/* Concurrent free page, same logic as sequential free page */
__global__ void LinkedList_freePage_Concurrent_kernel(int Nthreads, int *d_pageIdx, int *d_counts){
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if (tid>=Nthreads) return;
	for (int trial=0; trial<(NTRIALS+BURN); trial++){
		volatile unsigned long long start = (unsigned long long)clock();
		int pageID = d_pageIdx[trial*Nthreads+tid];
		volatile unsigned long long stop = (unsigned long long)clock();
		// save metrics
		d_counts[trial*Nthreads+tid] = (int)(stop - start);
	}

	// aggregate time (cycles) across trials, save to row 0 of d_counts
	unsigned long long Avg = 0;
	for (int trial=BURN; trial<(NTRIALS+BURN); trial++)
		Avg+=d_counts[trial*Nthreads+tid];
	Avg = Avg/NTRIALS;
	d_counts[tid] = (int)Avg;
}

/* Host function to run concurrent getpage and concurrent freepage*/
__host__ void LinkedList_Concurrent(int Nthreads){
	// allocate d_pageIdx and d_counts on GPU
	int *d_pageIdx, *d_counts, *h_counts;
	gpuErrchk (cudaMalloc((void**)&d_pageIdx, (NTRIALS+BURN)*Nthreads*sizeof(int)) );
	gpuErrchk (cudaMalloc((void**)&d_counts, (NTRIALS+BURN)*Nthreads*sizeof(int)) );
	h_counts = (int*)malloc((NTRIALS+BURN)*Nthreads*sizeof(int));

	// run get page kernel
	void* kernelArgs[] = {&Nthreads, &d_pageIdx, &d_counts};
	cudaLaunchCooperativeKernel((void*)&LinkedList_getPage_Concurrent_kernel, Nthreads/32, 32, (void**)kernelArgs);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );

	// aggregate time (cycles) and print out
	gpuErrchk( cudaMemcpy(h_counts, d_counts, 1*Nthreads*sizeof(int), cudaMemcpyDeviceToHost) );
	unsigned long long Avg = 0;
	for (int i=0; i<Nthreads; i++) Avg+=h_counts[i]; 
	Avg = Avg/Nthreads;
	printf("getPage,%d,%llu\n", Nthreads, Avg);


	// run free page kernel
	cudaLaunchCooperativeKernel((void*)&LinkedList_freePage_Concurrent_kernel, Nthreads/32, 32, (void**)kernelArgs);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );

	// aggregate time (cycles) and print out
	gpuErrchk( cudaMemcpy(h_counts, d_counts, 1*Nthreads*sizeof(int), cudaMemcpyDeviceToHost) );
	Avg = 0;
	for (int i=0; i<Nthreads; i++) Avg+=h_counts[i];
	Avg = Avg/Nthreads;
	printf("freePage,%d,%llu\n", Nthreads, Avg);

	// end
	cudaFree(d_pageIdx); cudaFree(d_counts);

}

/* Kernel to get 1 page with Random Walk, record step counts */
__global__ void LinkedList_get1page_kernel(int Nthreads, int *d_step_counts, int *d_pageIdx){
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if (tid<Nthreads){
		volatile clock_t start = clock();
		int step_counts;
		int pageID = getPageLinkedList(&step_counts);
		// freePageLinkedList(pageID);
		// pageID = getPageLinkedList(&step_counts);
		volatile clock_t stop = clock();
		d_step_counts[tid] = stop - start;
		// mem check
		void *page = pageAddress(pageID);
		fillPage(page);
		// save pageID to global mem
		d_pageIdx[tid] = pageID;
	}
}

/* Kernel to free 1 page with Random Walk, record step counts */
__global__ void LinkedList_free1page_kernel(int Nthreads, int *d_step_counts, int *d_pageIdx){
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	int pageID = d_pageIdx[tid];
	if (tid<Nthreads){
		volatile clock_t start = clock();
		int step_counts;
		freePageLinkedList(pageID, &step_counts);
		// freePageLinkedList(pageID);
		// pageID = getPageLinkedList(&step_counts);
		volatile clock_t stop = clock();
		d_step_counts[tid] = stop - start;
	}
}


/* Execute 2 types of kernels:
	1. sequential get page
	2. sequential free page
	3. concurrent get page
	4. concurrent free page
 */
void testLinkedList2(int Nthreads){
	// allocate pageIDx array on device
	int *d_pageIdx;
	gpuErrchk( cudaMalloc((void**)&d_pageIdx, (NTRIALS+BURN)*Nthreads*sizeof(int)) );
	// allocate metrics array on device
	int *d_metrics;
	gpuErrchk( cudaMalloc((void**)&d_metrics, (NTRIALS+BURN)*Nthreads*sizeof(int)) );

	// run sequential get page
	printf("==================== RUNNING SEQUENTIAL GET PAGE ======================\n");
	void* kernelArgs[] = {&Nthreads, &d_metrics, &d_pageIdx};
	cudaLaunchCooperativeKernel((void*)&LinkedList_getPage_Sequential_kernel, 1, 1, (void**)kernelArgs);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );

	// run sequential free page
	printf("==================== RUNNING SEQUENTIAL FREE PAGE ======================\n");
	cudaLaunchCooperativeKernel((void*)&LinkedList_freePage_Sequential_kernel, 1, 1, (void**)kernelArgs);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );

	cudaFree(d_pageIdx); cudaFree(d_metrics);

	// run concurrent get page & free page
	printf("==================== RUNNING CONCURRENT GET & FREE PAGE ======================\n");
	for (int i=32; i<=4992; i+=32)
		LinkedList_Concurrent(i);

}

/* Execute one kernel of N threads, each gets 1 page with Random Walk
	input: Nthreads: 		number of threads
	return: *avgStep:		average of step counts across all threads
			*avgMaxWarp:	average of Max of Warp across all warps
			*runTime:		total run time (s)
 */
Metrics_t testLinkedList(int Nthreads){
	// allocate metrics array on host
	int *h_step_counts = (int*)malloc(Nthreads*sizeof(int));
	// allocate metrics array on gpu
	int *d_step_counts;
	gpuErrchk( cudaMalloc((void**)&d_step_counts, Nthreads*sizeof(int)) );
	// storage to save pageID for each thread
	int *d_pageIdx;
	gpuErrchk( cudaMalloc((void**)&d_pageIdx, Nthreads*sizeof(int)) );

	// execute get kernel;
	printf("============================= SINGLE THREAD TESTING ============================\n");
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	LinkedList_get1page_kernel <<< 1,1 >>> (1, d_step_counts, d_pageIdx);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float total_time;
	cudaEventElapsedTime(&total_time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	// printNumPagesLeftLinkedList();
	// copy metrics to host
	gpuErrchk( cudaMemcpy(h_step_counts, d_step_counts, 1*sizeof(int), cudaMemcpyDeviceToHost) );
	// aggregate metrics and return
	Metrics_t out = aggregate_metrics(h_step_counts, 1);
	out.runTime = total_time;
	printf("average steps: %f\n", out.avgStep);


	printf("============================= ALL THREAD TESTING ============================\n");
	// cudaEvent_t start, stop;
	// cudaEventCreate(&start);
	// cudaEventCreate(&stop);
	// cudaEventRecord(start, 0);
	LinkedList_get1page_kernel <<< ceil((float)Nthreads/32),32 >>> (Nthreads, d_step_counts, d_pageIdx);
	// gpuErrchk( cudaPeekAtLastError() );
	// gpuErrchk( cudaDeviceSynchronize() );
	// cudaEventRecord(stop, 0);
	// cudaEventSynchronize(stop);
	// float total_time;
	// cudaEventElapsedTime(&total_time, start, stop);
	// cudaEventDestroy(start);
	// cudaEventDestroy(stop);
	// printNumPagesLeftLinkedList();
	// copy metrics to host
	gpuErrchk( cudaMemcpy(h_step_counts, d_step_counts, 1*sizeof(int), cudaMemcpyDeviceToHost) );
	// aggregate metrics and return
	out = aggregate_metrics(h_step_counts, Nthreads);
	// out.runTime = total_time;
	printf("average steps: %f\n", out.avgStep);

	free(h_step_counts); cudaFree(d_step_counts); cudaFree(d_pageIdx);
	return out;
}


int main(int argc, char const *argv[])
{
	gpuErrchk(cudaSetDevice(0));
	/* command descriptions */
	if(argc>1 && ((strncmp(argv[1], "-h", 2) == 0) || (strncmp(argv[1], "-help", 4) == 0))){
		fprintf(stderr, "USAGE: ./unitTestRandomWalk [options]\n");
		fprintf(stderr, "OPTIONS:\n");
		// fprintf(stderr, "\t -pn, --pageNum <pageNum>\n");
		// fprintf(stderr, "\t\t Total Pages, default is 1000000\n\n");

		fprintf(stderr, "\t -tn, --threadNum <threadsNum>\n");
		fprintf(stderr, "\t\t Total threads that are asking pages, default is 5000\n\n");

		// fprintf(stderr, "\t -lp, --leftPage <leftPageNum>\n");
		// fprintf(stderr, "\t\t Pages left in the system, default is all pages free \n\n");
	}

	/* parse options */
	int Nthreads=0;
	for (int i=0; i<argc; i++){
		if((strncmp(argv[i], "-tn", 3) == 0) || (strcmp(argv[i], "--threadNum") == 0))
			Nthreads = atoi(argv[i]);

	}
	if (Nthreads==0) Nthreads = 5000;


	/* initialize system, all pages free, parameters defined in parallelPage.cuh */
	fprintf(stderr, "initializing page system ... \n");
	initPagesLinkedList();
	printNumPagesLeftLinkedList();

	/* repeat getpage with Random Walk */
	fprintf(stderr, "unit test with Total Pages = %d, Nthreads = %d ...\n", TOTAL_N_PAGES, Nthreads);
	// int AvailablePages = TOTAL_N_PAGES;
	// printf("T,N,A,Average_steps,Average_Max_Warp,Time(ms)\n");
	// for (int i=0; i<TOTAL_N_PAGES/Nthreads; i++){
	// 	// run kernel to get 1 page for each thread
	// 	Metrics_t metrics = testLinkedList(Nthreads);
	// 	// print results to stdout
	// 	printf("%d,%d,%d,%f,%f,%f\n", TOTAL_N_PAGES, Nthreads, AvailablePages, metrics.avgStep, metrics.avgMaxWarp, metrics.runTime);
	// 	AvailablePages-=Nthreads;
	// }
	testLinkedList2(Nthreads);

	return 0;
}