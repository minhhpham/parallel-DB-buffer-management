#include "../source/CollabRW_BM.cuh"
#include "metrics.h"
#include <iostream>
#include <vector>
using namespace std;

#define NTHREADS 10000
#define NITERATIONS 100
#define NSAMPLES 1

typedef struct PageData_t
{
    int threadId;
    int pageId;
    int iteration;
    int sample;
} PageData_t;


__global__ void get1page_kernel(int Nthreads, int *pageIds, int *d_step_counts){
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if (tid<Nthreads){
		int step_counts;
		int *tmp = d_step_counts? &step_counts : 0;
		int pageID = getPage(tmp);
		if (d_step_counts) d_step_counts[tid] = step_counts;
        pageIds[tid] = pageID;
	}
}


vector<PageData_t> collect_one_sample(int sampleId){
    vector<PageData_t> out;
    int *d_pageIds;   // array of pageIds from kernel execution
    gpuErrchk( cudaMalloc(&d_pageIds, NTHREADS*sizeof(int)) );
    int *h_pageIds = (int*)malloc(NTHREADS*sizeof(int));
    resetMemoryManager();

    for (int i=0; i<NITERATIONS; i++){
        get1page_kernel <<< ceil((float)NTHREADS/32), 32 >>> (NTHREADS, d_pageIds, nullptr);
        gpuErrchk( cudaPeekAtLastError() );
	    gpuErrchk( cudaDeviceSynchronize() );
        cudaMemcpy(h_pageIds, d_pageIds, NTHREADS*sizeof(int), cudaMemcpyDeviceToHost);
        for (int tid=0; tid<NTHREADS; tid++){
            out.push_back({
                .threadId = tid,
                .pageId = h_pageIds[tid],
                .iteration = i,
                .sample = sampleId
            });
            h_pageIds[tid] = {};
        }
    }

    cudaFree(d_pageIds);
    free(h_pageIds);
    return out;
}





int main(int argc, char const *argv[])
{
    initMemoryManagement(1, 1024);

    fprintf(stdout, "threadId,pageId,iteration,sampleId\n");
    for (int sampleId=0; sampleId<NSAMPLES; sampleId++){
        auto sample = collect_one_sample(sampleId);
        for (auto point:sample)
            fprintf(stdout, "%d,%d,%d,%d\n", point.threadId, point.pageId, point.iteration, sampleId);
    }

    return 0;
}

