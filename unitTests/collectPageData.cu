#include "../source/CollabRW_BM.cuh"
#include "metrics.h"
#include <iostream>
#include <vector>
using namespace std;

#define NTHREADS 32
#define NSAMPLES 10000

typedef struct PageData_t
{
    int threadId;
    int pageId;
    int iteration;
    int sample;
} PageData_t;


__global__ void get1page_kernel(int Nthreads, int *pageIds, int *d_step_counts, bool collectWarpData=false){
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if (tid<Nthreads){
		int step_counts;
		int *tmp = d_step_counts? &step_counts : 0;
		int pageID = getPage(tmp, collectWarpData);
		if (d_step_counts) d_step_counts[tid] = step_counts;
        pageIds[tid] = pageID;
	}
}


void collect_one_sample(int sampleId){
    int *d_pageIds;   // array of pageIds from kernel execution
    gpuErrchk( cudaMalloc(&d_pageIds, NTHREADS*sizeof(int)) );

    get1page_kernel <<< ceil((float)NTHREADS/32), 32 >>> (NTHREADS, d_pageIds, nullptr, true);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );


    cudaFree(d_pageIds);
}





int main(int argc, char const *argv[])
{
    initMemoryManagement(2, 2147);

    for (int sampleId=0; sampleId<NSAMPLES; sampleId++){
        resetMemoryManager();
        prefillBuffer(0.1);
        collect_one_sample(sampleId);
    }

    return 0;
}

