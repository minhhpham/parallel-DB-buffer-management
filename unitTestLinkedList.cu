#include "parallelPage.cuh"
#include "metrics.h"

static __device__ void fillPage(void *page){
	char *ptr = (char*)page;
	for (int i=0; i<PAGE_SIZE; i++)
		ptr[i] = 1;
}

/* Kernel to get 1 page with Random Walk, record step counts */
__global__ void LinkedList_get1page_kernel(int Nthreads, int *d_step_counts){
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if (tid<Nthreads){
		int step_counts;
		int pageID = getPageLinkedList(&step_counts);
		// freePageLinkedList(pageID);
		// pageID = getPageLinkedList(&step_counts);
		// d_step_counts[tid] = step_counts;
		// mem check
		void *page = pageAddress(pageID);
		fillPage(page);
	}
}


/* Execute one kernel of N threads, each gets 1 page with Random Walk
	input: Nthreads: 		number of threads
	return: *avgStep:		average of step counts across all threads
			*avgMaxWarp:	average of Max of Warp across all warps
			*runTime:		total run time (s)
 */
Metrics_t runLinkedList(int Nthreads){
	// allocate metrics array on host
	int *h_step_counts = (int*)malloc(Nthreads*sizeof(int));
	// allocate metrics array on gpu
	int *d_step_counts;
	gpuErrchk( cudaMalloc((void**)&d_step_counts, Nthreads*sizeof(int)) );

	// execute kernel;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	LinkedList_get1page_kernel <<< ceil((float)Nthreads/32), 32 >>> (Nthreads, d_step_counts);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float total_time;
	cudaEventElapsedTime(&total_time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printNumPagesLeftLinkedList();

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
	int AvailablePages = TOTAL_N_PAGES;
	printf("T,N,A,Average_steps,Average_Max_Warp,Time(ms)\n");
	for (int i=0; i<TOTAL_N_PAGES/Nthreads; i++){
		// run kernel to get 1 page for each thread
		Metrics_t metrics = runLinkedList(Nthreads);
		// print results to stdout
		printf("%d,%d,%d,%f,%f,%f\n", TOTAL_N_PAGES, Nthreads, AvailablePages, metrics.avgStep, metrics.avgMaxWarp, metrics.runTime);
		AvailablePages-=Nthreads;
	}

	return 0;
}