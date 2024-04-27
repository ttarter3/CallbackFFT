#ifndef AXPY_CUH
#define AXPY_CUH

// Standard Headers
#include <stdio.h>

template<typename T>
__global__
void saxpy(T a, T *x, T *y, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
         i < n; 
         i += blockDim.x * gridDim.x) 
      {
        T p_val = 0.0; int N = 1000;
        for (int xx = 0; xx < N; xx++) {
           p_val += x[i];
        }
        for (int xx = 0; xx < N - 1; xx++) {
          p_val -= x[i];
        }
        y[i] += a * x[i];
      }
}

#endif //AXPY_CUH
