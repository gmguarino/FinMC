#include <vector>

#include <time.h>
#include <math.h>
#include <cuda_runtime.h>

#ifndef _KERNEL_CUH_
#define _KERNEL_CUH_


/* 
Wrapper function calling a CUDA kernel to perform a monte carlo simulation
of the variation in stock prices.
 */
void mc_call(
    float *device_array,
    float time_period,
    float k,
    float barrier_level,
    float initial_price,
    float volatility,
    float drift,
    float rate,
    float t_step,
    float *device_random_array,
    unsigned n_steps,
    unsigned n_simulations,
    const dim3 blockShape,
    const dim3 gridShape
);

#endif