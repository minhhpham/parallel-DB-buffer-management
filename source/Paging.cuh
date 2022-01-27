#ifndef MM_PAGING_CUH
#define MM_PAGING_CUH

#define TOTAL_N_PAGES_DEFAULT 	41900000
#define PAGE_SIZE_DEFAULT		264	

#include "ErrorHandler.cuh"
#include <stdio.h>

static int h_total_n_pages;
static __device__ int d_total_n_pages;
static __device__ int d_pageSize;

#define GROUP_N_PAGES   100000  // group pages for faster initialization
                                // (initialize a single big chunk with the CUDA API may fail)
static __device__ void **d_page_groups;    // actual address to pages, (TOTAL_N_PAGES/GROUP_N_PAGES) groups, each with GROUP_N_PAGES pages

__device__ void *pageAddress(int pageID){
    // find group id
    int groupID = pageID/GROUP_N_PAGES;
    // find offset on group
    int offset = pageID - groupID*GROUP_N_PAGES;
    // calculate actual address
    void *groupAddr = d_page_groups[groupID];
    return (void*)((char*)groupAddr + offset*d_pageSize);
}


/* initialize actual memory pages 
    - init pages by groups (initialize a single big chunk with the CUDA API may fail)
    - allocate an array of page map and transfer the array of group pointers to this array
    - set the device global variables d_total_n_pages and d_pageSize
 */
__host__ void initPages(int nPages=TOTAL_N_PAGES_DEFAULT, int pageSize=PAGE_SIZE_DEFAULT){
    // init pages by groups (initialize a single big chunk with the CUDA API may fail)
    int n_groups = ceil((float)nPages/GROUP_N_PAGES);
    fprintf(stderr, "initializing memory pages on GPU ... Total Size = %'lld MB \n", (long long)nPages*pageSize/1048576);
    // initialize (TOTAL_N_PAGES/GROUP_N_PAGES) groups on GPU memory, save their pointers on a host array
    void **h_groups = (void**)malloc(n_groups*sizeof(void*));
    for (int i=0; i<n_groups; i++){ // allocate each group
        gpuErrchk( cudaMalloc((void**)&h_groups[i], GROUP_N_PAGES*pageSize) );
    }
    
    // allocate an array for page map and transfer the array of group pointers to this array
    void* tmp;
    cudaMalloc(&tmp, n_groups*sizeof(void*));
    gpuErrchk( cudaMemcpy(tmp, h_groups, n_groups*sizeof(void*), cudaMemcpyHostToDevice) );
    // set d_page_groups to this array
    gpuErrchk( cudaMemcpyToSymbol(d_page_groups, &tmp, sizeof(void*)) );
    
    // set the device global variables d_total_n_pages and d_pageSize
    h_total_n_pages = nPages;
    gpuErrchk( cudaMemcpyToSymbol(d_total_n_pages, &nPages, sizeof(int)) );
    gpuErrchk( cudaMemcpyToSymbol(d_pageSize, &pageSize, sizeof(int)) );

    // end
    free(h_groups);
}




#endif