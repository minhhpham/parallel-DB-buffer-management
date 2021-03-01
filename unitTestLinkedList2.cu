#include <stdio.h>

__global__ void LinkedList_atomic_kernel(int Nthreads, int *d_lock, unsigned long long *d_time){
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if (tid<Nthreads){
		int lock = 1;
		volatile clock_t start = clock();
		while (lock){
			lock = atomicExch(d_lock, 1);
			if (lock==0) atomicExch(d_lock, 0);
		}
		volatile clock_t stop = clock();
		d_time[tid] = (unsigned long long) (stop - start);
		// printf("%llu\n", d_time[tid]);
	}
}


/* Execute one kernel of N threads, each gets 1 page with Random Walk
	input: Nthreads: 		number of threads
	return: *avgStep:		average of step counts across all threads
			*avgMaxWarp:	average of Max of Warp across all warps
			*runTime:		total run time (s)
 */
unsigned long long runLinkedList(int Nthreads){
	int *d_lock;
	unsigned long long *d_time;
	cudaMalloc((void**)&d_lock, sizeof(int));
	cudaMemset(d_lock, 0, sizeof(int));
	cudaMalloc((void**)&d_time, Nthreads*sizeof(unsigned long long));
	LinkedList_atomic_kernel <<< ceil((float)Nthreads/32), 32 >>> (Nthreads, d_lock, d_time);
	cudaPeekAtLastError();
	cudaDeviceSynchronize();

	unsigned long long *h_time = (unsigned long long*)malloc(Nthreads*sizeof(unsigned long long));
	cudaMemcpy(h_time, d_time, Nthreads*sizeof(unsigned long long), cudaMemcpyDeviceToHost);
	unsigned long long avg = 0;
	for (int i=0; i<Nthreads; i++) 
		avg+=h_time[i];
	avg = avg/Nthreads;

	cudaFree(d_lock); cudaFree(d_time); free(h_time);
	return avg;
}


int main(int argc, char const *argv[])
{
	/* repeat getpage with Random Walk */
	for (int Nthreads=1; Nthreads<5000; Nthreads+=50){
		for (int burn=0; burn<100; burn++)
			runLinkedList(Nthreads);

		unsigned long long time[100];
		for (int trial=0; trial<100; trial++){
			time[trial] = runLinkedList(Nthreads);
			if (trial>0 && time[trial>time[trial-1]*1.5])
				time[trial] = time[trial-1];
		}

		unsigned long long avg = 0;
		for (int i=0; i<100; i++)
			avg+=time[i];
		avg = avg/100;

		printf("%d,%llu\n", Nthreads, avg);
	}

	return 0;
}