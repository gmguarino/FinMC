#include "mc_kernel.hh"


__global__ void mc_kernel(
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
    unsigned n_simulations
)
{
    /* Define the thread, block and grid indices organised such that all the threads
    in a block compute sequential elements of the device array. */

    const unsigned tid =  threadIdx.x + threadIdx.y * blockDim.x;
    const unsigned bid = blockIdx.x + blockIdx.y * gridDim.x;
    const unsigned t_per_block = blockDim.x * blockDim.y;

    size_t array_idx = tid + bid * t_per_block;
    size_t random_array_idx = tid + bid * t_per_block;

    float current_value = initial_price;
    if (array_idx < n_simulations)
    {
        for (size_t n = 0; n < n_steps; n++)
        {
            current_value = current_value * (1 + drift * t_step + volatility * device_random_array[random_array_idx * n_steps + n]);
            if (current_value <= barrier_level)
            {
                break;
            }
        }
        float payoff = (current_value>k ? current_value - k: 0);
        __syncthreads();
        device_array[array_idx] = exp(-rate * time_period) * payoff;
    }
}

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
)
{
    mc_kernel<<<gridShape, blockShape>>>(
        device_array,
        time_period,
        k,
        barrier_level,
        initial_price,
        volatility,
        drift,
        rate,
        t_step,
        device_random_array,
        n_steps,
        n_simulations
    );
}