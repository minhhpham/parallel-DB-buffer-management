#include "parallelPage.cuh"

__global__ void writeKernel(int pageID){
	// write 4-byte integers from 1 to PAGE_SIZE/4 to each page
	int *arr = (int*)pageAddress(pageID);
	for (int i=0; i<PAGE_SIZE/4; i++){
		arr[i] = i+1;
	}
}

__global__ void readKernel(int pageID){
	// check if the integers on the page are consistent with the write
	int *arr = (int*)pageAddress(pageID);
	for (int i=0; i<PAGE_SIZE/4; i++){
		if (arr[i] != (i+1))
			__trap();
	}
}

int main(int argc, char const *argv[])
{
	/* init Buffer */
	initPagesRandomWalk_BM();
	/* test reading and writing on all pages */
	for (int pageID=0; pageID<TOTAL_N_PAGES; pageID++){
		fprintf(stderr, "\r%8d", pageID);
		writeKernel <<< 1, 1 >>> (pageID);
		readKernel <<< 1, 1 >>> (pageID);
		gpuErrchk( cudaPeekAtLastError() );
		gpuErrchk( cudaDeviceSynchronize() );
	}

	return 0;
}