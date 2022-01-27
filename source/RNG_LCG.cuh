#ifndef RNG_LCG_CUH
#define RNG_LCG_CUH

/* function to replace curand, better efficiency */

#define LCG_M 1<<31
#define LCG_A 1103515245
#define LCG_C 12345
__device__ static inline int RNG_LCG(int seed){
    long long seed_ = (long long)seed;
    return (int)((LCG_A*seed_ + LCG_C)%(LCG_M));
}


#endif