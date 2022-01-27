#include "../source/Paging.cuh"
#include <iostream>

__global__ void writeKernel(int pageID){
	// write 4-byte integers from 1 to PAGE_SIZE/4 to each page
	int pageSize = PAGE_SIZE_DEFAULT;
	int *arr = (int*)pageAddress(pageID);
	for (int i=0; i<pageSize/4; i++){
		arr[i] = i+1;
	}
}

__global__ void readKernel(int pageID){
	// check if the integers on the page are consistent with the write
	int *arr = (int*)pageAddress(pageID);
	int pageSize = PAGE_SIZE_DEFAULT;
	for (int i=0; i<pageSize/4; i++){
		if (arr[i] != (i+1))
			__trap();
	}
}

int main(int argc, char const *argv[])
{
	/* init Buffer */
	initPages();
	/* test reading and writing on all pages */
	fprintf(stderr, "testing ... \n");
	for (int pageID=0; pageID<TOTAL_N_PAGES_DEFAULT; pageID++){
		writeKernel <<< 1, 1 >>> (pageID);
		readKernel <<< 1, 1 >>> (pageID);
		gpuErrchk( cudaPeekAtLastError() );
		gpuErrchk( cudaDeviceSynchronize() );

		// display progress
		std::cerr << "[";
		int barWidth = 70;
		float progress = (float)pageID/TOTAL_N_PAGES_DEFAULT;
		int pos = barWidth * progress;
		for (int i = 0; i < barWidth; ++i) {
			if (i < pos) std::cerr << "=";
			else if (i == pos) std::cerr << ">";
			else std::cerr << " ";
		}
		std::cerr << "] " << int(progress * 100.0) << " %\r";
		std::cerr.flush();
	}

	fprintf(stderr, "done \n");

	return 0;
}