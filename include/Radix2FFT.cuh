#ifndef INCLUDE_RADIX2FFT_CUH
#define INCLUDE_RADIX2FFT_CUH

#include <cuda_runtime.h>
#include <math.h>

// Define constants
#define BLOCK_SIZE 32
#define PI 3.14159265358979323846

// Define constant memory for twiddle factors
__constant__ float2 twiddle_factors[BLOCK_SIZE];

// Radix-2 Cooley-Tukey FFT
template<typename T>
__global__ void Radix2FFT(T* data, int N) {
    // Shared memory for data
    __shared__ T shared_data[BLOCK_SIZE];

    // Load data into shared memory
    int tid = threadIdx.x;
    int idx = blockIdx.x * BLOCK_SIZE + tid;
    if (idx < N)
        shared_data[tid] = data[idx];
    __syncthreads();

    // Perform radix-2 FFT
    for (int s = 1; s < BLOCK_SIZE; s *= 2) {
        int stride = s * 2;
        for (int i = tid; i < BLOCK_SIZE; i += stride) {
            int even_idx = i;
            int odd_idx = i + s;

            T even = shared_data[even_idx];
            T odd = shared_data[odd_idx];

            T twiddle_factor = twiddle_factors[i * (BLOCK_SIZE / stride)];

            T t = make_float2(odd.x * twiddle_factor.x - odd.y * twiddle_factor.y,
                                   odd.x * twiddle_factor.y + odd.y * twiddle_factor.x);

            shared_data[even_idx] = make_float2(even.x + t.x, even.y + t.y);
            shared_data[odd_idx] = make_float2(even.x - t.x, even.y - t.y);
        }
        __syncthreads();
    }

    // Write result back to global memory
    if (idx < N)
        data[idx] = shared_data[tid];
}

#endif // INCLUDE_RADIX2FFT_CUH