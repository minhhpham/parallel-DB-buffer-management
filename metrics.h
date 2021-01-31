typedef struct 
{
	float avgStep;
	float avgMaxWarp;
	float runTime;	// in ms
} Metrics_t;

/* step_counts; array of step counts for eac thread */
static inline Metrics_t aggregate_metrics(int *step_counts, int Nthreads){
	Metrics_t out;
	// calculate avg of step counts 
	unsigned long Sum_Step = 0;
	for (int i=0; i<Nthreads; ++i)
		Sum_Step+=step_counts[i];
	out.avgStep = (float)Sum_Step/Nthreads;
	
	// calculate avg of max warp
	unsigned long Sum_MaxWarp = 0; int max_warp = 0;
	for (int i=0; i<Nthreads; ++i){
		if (i%32==0) max_warp = 0;	// reset for new warp
		if (step_counts[i]>max_warp) max_warp = step_counts[i];
		if (i%32==31 || i==Nthreads-1)	// end of warp, write result
			Sum_MaxWarp+=max_warp;
	}
	int Nwarps = ceil((float)Nthreads/32);
	out.avgMaxWarp = (float)Sum_MaxWarp/Nwarps;	
	return out;
}