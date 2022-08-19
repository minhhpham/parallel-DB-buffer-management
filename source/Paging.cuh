#ifndef MM_PAGING_CUH
#define MM_PAGING_CUH

#define PAGE_SIZE_DEFAULT		512	
#define TOTAL_SIZE_GB_DEFAULT   8
#define GB_SIZE                 1024ull*1024ull*1024ull
#define TOTAL_N_PAGES_DEFAULT 	(TOTAL_SIZE_GB_DEFAULT)*(GB_SIZE)/(PAGE_SIZE_DEFAULT)

#include "ErrorHandler.cuh"
#include <stdio.h>
#include <stdint.h>

/* ------------------- DECLARATION AND USAGE ------------------- */
/* return page address from pageID */
__device__ void *pageAddress(int pageID);   // get page address from page ID
__device__ int getPageID(void *pageAddr);   // get pageID from address

/* initialize the paging system, return -1 if invalid */
__host__ void initPages(int nGB=TOTAL_SIZE_GB_DEFAULT, int pageSize=PAGE_SIZE_DEFAULT);

/* ------------------- IMPLEMENTATION --------------------------*/

static int h_total_n_pages;             // num of pages
static __device__ int d_total_n_pages;  // num of pages device variable
static __device__ int d_pageSize;
static __device__ void **d_page_groups;    // actual address to pages, (nGB) groups, each with (GB_SIZE)/(pageSize) pages
static __device__ int d_nPages_per_group;  // number of pages per GB group
static __device__ int d_nGB;            // number of GB allocated in the system

__device__ void *pageAddress(int pageID){
    // find group id
    int groupID = pageID/d_nPages_per_group;
    // find offset on group
    int offset = pageID - groupID*d_nPages_per_group;
    // calculate actual address
    void *groupAddr = d_page_groups[groupID];
    return (void*)((char*)groupAddr + offset*d_pageSize);
}

__device__ int getPageID(void *pageAddr){
    // find group id by looking up the ranges on each group
    int gid;
    #pragma unroll
    for (gid=0; gid<d_nGB; gid++){
        uint64_t target = (uint64_t)pageAddr;
        uint64_t lower = (uint64_t)d_page_groups[gid];
        uint64_t upper = (uint64_t)(lower + GB_SIZE - 1);
        if (lower<=target && target<=upper)
            return gid*d_nPages_per_group + ( (target-lower)/d_pageSize );
    }
    return -1;
}

/* initialize actual memory pages 
    - init pages by groups (initialize a single big chunk with the CUDA API may fail)
    - allocate an array of page map and transfer the array of group pointers to this array
    - set the device global variables d_total_n_pages and d_pageSize
 */
__host__ void initPages(int nGB, int pageSize){
    // init pages by groups (initialize a single big chunk with the CUDA API may fail)
    fprintf(stderr, "initializing memory pages on GPU ... Total Size = %'lld MB \n", (long long)nGB*GB_SIZE/1048576);
    // initialize (nGB) groups on GPU memory, each of 1GB, save their pointers on a host array
    void **h_groups = (void**)malloc(nGB*sizeof(void*));
    for (int i=0; i<nGB; i++){ // allocate each GB
        gpuErrchk( cudaMalloc((void**)&h_groups[i], GB_SIZE) );
    }
    
    // allocate an array for page map and transfer the array of group pointers to this array
    void* tmp;
    cudaMalloc(&tmp, nGB*sizeof(void*));
    gpuErrchk( cudaMemcpy(tmp, h_groups, nGB*sizeof(void*), cudaMemcpyHostToDevice) );
    // set d_page_groups to this array
    gpuErrchk( cudaMemcpyToSymbol(d_page_groups, &tmp, sizeof(void*)) );
    
    // set the device global variables d_total_n_pages and d_pageSize and d_nPages_per_group and d_nGB
    h_total_n_pages = nGB*GB_SIZE/pageSize;
    gpuErrchk( cudaMemcpyToSymbol(d_total_n_pages, &h_total_n_pages, sizeof(int)) );
    gpuErrchk( cudaMemcpyToSymbol(d_pageSize, &pageSize, sizeof(int)) );
    int nPages_per_group = (int)(GB_SIZE/pageSize);
    gpuErrchk( cudaMemcpyToSymbol(d_nPages_per_group, &nPages_per_group, sizeof(int)) );
    gpuErrchk( cudaMemcpyToSymbol(d_nGB, &nGB, sizeof(int)) );

    // end
    free(h_groups);
}




#endif