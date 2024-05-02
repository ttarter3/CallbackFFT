#ifndef RADIX2KERNEL_CUH
#define RADIX2KERNEL_CUH

#include "CuMath.cuh"
#include "Define.h"

#include <cuda_runtime.h>
#include <cooperative_groups.h>

// Standard Headers
#include <stdio.h>
#include <math.h>
#include "cuComplex.h"
#include <cmath>
#include <assert.h>


#define tx threadIdx.x
#define ty threadIdx.y
#define tz threadIdx.z
#define bx blockIdx.x
#define by blockIdx.y
#define bz blockIdx.z

#define PRINTIDX 6
#define X 32
#define PI 3.141592

namespace cg = cooperative_groups;

__global__ void Radix2ShiftIgnore(cuComplex* in,const unsigned int N, unsigned int M)
{
  for (int ii = bx * blockDim.x + tx;
       ii < N;
       ii += blockDim.x * gridDim.x) {
    unsigned int swap_idx = ii;

    unsigned int temp = 0;
    for (int jj = 0; jj < X; jj++) {
      temp |= ((swap_idx >> (jj + 0)) & 0x01) << (X - 1 - jj);
    }
    swap_idx = temp >> (X - M);

    if (ii < swap_idx) {
      Swap(in[swap_idx], in[ii]);
    }
  }
}

__global__ void Radix2MultIgnore(cuComplex* in, const unsigned int N, unsigned int M)
{
  unsigned int x[SIZE2];
  cuComplex y[SIZE2];

  for (int ii = bx * blockDim.x + tx;
       ii < N / SIZE2;
       ii += blockDim.x * gridDim.x) {
    for (int jj = 0; jj < SIZE2; jj++) {
      x[jj] = (ii / M) * SIZE2 * M + ii % M + jj * M;
    }

    float angle = -2 * M_PI * ((N / (M * SIZE2)) * ii - (N / SIZE2) * (ii / M)) / N;
    cuComplex weight = make_cuComplex(cos(angle), sin(angle));

//    if (bx == 1 && tx == 1) {
//      printf("#%d(%.12f, %.12f)->%d(%d) %.12f\t %d(%d) %.12f\n", M, weight.x, weight.y, x[0], x[0] % TILE_SIZE, in[x[0]].x, x[1], x[1] % TILE_SIZE, in[x[1]].x);
//    }

    // if (bx == 0 && M == 1) {
    //   printf("%d %d %d\n", M, tx, x[0]);
    // }

    y[0] = in[x[0]];
    y[1] = Mult(in[x[1]], weight);

    in[x[0]] = Add(y[0], y[1]);
    in[x[1]] = Sub(y[0], y[1]);

//    if (bx == 1 && tx == 1) {
//      printf("#%d->%d(%d) %.12f %.12f\t %d(%d) %.12f %.12f\n", M, x[0], x[0] % TILE_SIZE, y[0].x, in[x[0]].x, x[1], x[1] % TILE_SIZE, y[1].x, in[x[1]].x);
//    }
  }
  __syncthreads();
}



__global__ void Radix2Shift(cuComplex* in,const unsigned int N, unsigned int M)
{

  __shared__ cuComplex shared_data[TILE_SIZE];
  for (int ii = bx * blockDim.x + tx;
       ii < N;
       ii += blockDim.x * gridDim.x) {
    unsigned int swap_idx = ii;

    unsigned int temp = 0;
    for (int jj = 0; jj < X; jj++) {
      temp |= ((swap_idx >> (jj + 0)) & 0x01) << (X - 1 - jj);
    }
    swap_idx = temp >> (X - M);

    if (ii < swap_idx) {
      shared_data[ii % TILE_SIZE] = in[ii];
      in[ii] = in[swap_idx];
      in[swap_idx] = shared_data[ii % TILE_SIZE];

      // Swap(in[swap_idx], in[ii]);
    }
  }
}

__global__ void Radix2Mult(cuComplex* in, const unsigned int N, unsigned int M)
{
  unsigned int x[SIZE2];
  cuComplex y[SIZE2];

  assert(bx * blockDim.x + tx < N / SIZE2);

  for (int ii = bx * blockDim.x + tx;
       ii < N / SIZE2;
       ii += blockDim.x * gridDim.x) {
    for (int jj = 0; jj < SIZE2; jj++) {
      x[jj] = (ii / M) * SIZE2 * M + ii % M + jj * M;
    }

    float angle = -2 * M_PI * ((N / (M * SIZE2)) * ii - (N / SIZE2) * (ii / M)) / N;
    cuComplex weight = make_cuComplex(cos(angle), sin(angle));

//    if (bx == 1 && tx == 1) {
//      printf("#%d(%.12f, %.12f)->%d(%d) %.12f\t %d(%d) %.12f\n", M, weight.x, weight.y, x[0], x[0] % TILE_SIZE, in[x[0]].x, x[1], x[1] % TILE_SIZE, in[x[1]].x);
//    }

    // if (bx == 0 && M == 1) {
    //   printf("%d %d %d\n", M, tx, x[0]);
    // }

    y[0] = in[x[0]];
    y[1] = Mult(in[x[1]], weight);

    in[x[0]] = Add(y[0], y[1]);
    in[x[1]] = Sub(y[0], y[1]);

  //  if (bx == 1 && tx == 1) {
  //    printf("#%d->%d(%d) %.12f %.12f\t %d(%d) %.12f %.12f\n", M, x[0], x[0] % TILE_SIZE, y[0].x, in[x[0]].x, x[1], x[1] % TILE_SIZE, y[1].x, in[x[1]].x);
  //  }
  }
  __syncthreads();
}

// __global__ void Radix2(cuComplex* in, const unsigned int N, unsigned int M) {
//   Radix2Shift(in, N, M);
//   for (int k = 1; k <  N; k *= SIZE2) {
//     Radix2Mult(in, N, k);
//   }
// }


__global__ void Radix2Mult2XThread(cuComplex* in, const unsigned int N, unsigned int M)
{
  __shared__ cuComplex shared_data[TILE_SIZE * SIZE2];
  __shared__ unsigned int x[TILE_SIZE * SIZE2];

  for (int ii = bx * blockDim.x + tx;
       ii < N;
       ii += blockDim.x * gridDim.x) {
        
    x[ii % TILE_SIZE * SIZE2] = (ii / 2 / M) * SIZE2 * M + (ii / 2) % M + (ii & 0x1) * M;

    float angle = (ii & 0x1) * -2 * M_PI * ((N / (M * SIZE2)) * ii / 2 - (N / SIZE2) * (ii / 2 / M)) / N;
    cuComplex weight = make_cuComplex(cos(angle), sin(angle));

    shared_data[ii % TILE_SIZE * SIZE2] = Mult(in[x[ii % TILE_SIZE * SIZE2]], weight);

    if (ii & 0x1 == 0) 
      in[x[ii % TILE_SIZE * SIZE2]] = Add(shared_data[ii & 0xFFFFFFFE % TILE_SIZE * SIZE2], shared_data[ii & 0xFFFFFFFE % TILE_SIZE * SIZE2 + 1]);
    else 
      in[x[ii % TILE_SIZE * SIZE2]] = Sub(shared_data[ii & 0xFFFFFFFE % TILE_SIZE * SIZE2], shared_data[ii & 0xFFFFFFFE % TILE_SIZE * SIZE2 + 1]);

  }
  __syncthreads();
}



__global__ void Radix2MultShared(cuComplex* in, const unsigned int N, unsigned int M)
{
  unsigned int x[SIZE2];
  cuComplex y[SIZE2];
  __shared__ cuComplex shared_data[TILE_SIZE * SIZE2];

  cuComplex weight[SIZE2];
  weight[0] = make_cuComplex(1,0);

  for (int ii = bx * blockDim.x + tx;
       ii < N / SIZE2;
       ii += blockDim.x * gridDim.x) {

    float angle = -2 * M_PI * ((N / (M * SIZE2)) * ii - (N / SIZE2) * (ii / M)) / N;
    weight[1] = make_cuComplex(cos(angle), sin(angle));

    for (int jj = 0; jj < SIZE2; jj++) {
      x[jj] = (ii / M) * SIZE2 * M + ii % M + jj * M;
    }

    shared_data[tx] = Mult(in[x[tx % 2]], weight[tx % 2]);
    __syncthreads();
//    y[0] = in[x[0]];
//    y[1] = Mult(in[x[1]], weight);

    in[x[0]] = Add(y[0], y[1]);
    in[x[1]] = Sub(y[0], y[1]);
  }
  __syncthreads();
}



__global__ void Radix2Mult1st(cuComplex* in, const unsigned int N, unsigned int M)
{
  unsigned int x[SIZE2];
  cuComplex y[SIZE2];
  __shared__ cuComplex shared_data[TILE_SIZE * SIZE2];

  for (int ii = bx * blockDim.x + tx;
       ii < N / SIZE2;
       ii += blockDim.x * gridDim.x) {

      for (int jj = 0; jj <= SIZE2; jj++) {
        shared_data[2 * tx + jj] = in[2 * ii + jj];
      }
      __syncthreads();

      for (M = M ; M <= TILE_SIZE; M *= SIZE2) {
        for (int jj = 0; jj < SIZE2; jj++) {
          x[jj] = (ii / M) * SIZE2 * M + ii % M + jj * M;
        }

        float angle = -2 * M_PI * ((N / (M * SIZE2)) * ii - (N / SIZE2) * (ii / M)) / N;
        cuComplex weight = make_cuComplex(cos(angle), sin(angle));

        y[0] =      shared_data[x[0] % (SIZE2 * TILE_SIZE)];
        y[1] = Mult(shared_data[x[1] % (SIZE2 * TILE_SIZE)], weight);

        shared_data[x[0] % (SIZE2 * TILE_SIZE)] = Add(y[0], y[1]);
        shared_data[x[1] % (SIZE2 * TILE_SIZE)] = Sub(y[0], y[1]);
        __syncthreads();
      }
      
      in[x[0]] = shared_data[x[0] % (SIZE2 * TILE_SIZE)];
      in[x[1]] = shared_data[x[1] % (SIZE2 * TILE_SIZE)];

      __syncthreads();
  }
}

extern "C" __global__ void Radix2Mult2nd(cuComplex* in, const unsigned int N, unsigned int M , int coalescence)
{
  unsigned int x[MAX_COALESCENCE];
  bool valid[MAX_COALESCENCE];
  unsigned int x_1;
  cuComplex y[SIZE2];

  cg::grid_group grid = cg::this_grid();
  cg::thread_group thread = cg::this_thread_block();
  
  for (int xx = (bx * coalescence) * blockDim.x + tx;
       xx < N / SIZE2;
       xx += blockDim.x * gridDim.x) {
    
    int ii, jj;
    // for (jj = 0; jj < coalescence; jj++) {
    //   ii = (bx * coalescence + jj) * blockDim.x + tx;
    //   x[jj] = (ii / M) * SIZE2 * M + ii % M;

    //   valid[jj] = true;

    //   // if (ii >= N / 2) {
    //   //   printf("UC: %d\n", ii);
    //   // }
    // }
    // cg::sync(grid);

    for (M = M ; M < N; M *= SIZE2) {
      for (jj = 0; jj < coalescence; jj++) {
        ii = (bx * coalescence + jj) * blockDim.x + tx;
        // if (bx == 392 && tx == 1 && jj == 0) {
        //   printf("%d %d %d %d %d %d %d\n", (ii / M) * SIZE2 * M + ii % M + 0 * M, (ii / M) * SIZE2 * M + ii % M + 1 * M, bx, coalescence, jj, blockDim.x, tx);
        // }

        // if (x[jj] != (ii / M) * SIZE2 * M + ii % M + 0 * M) {
        //   // valid[jj] = false;
        //   printf("%d %d %d %d %d %d\n", M, x[jj], (ii / M) * SIZE2 * M + ii % M + 0 * M, bx, jj, tx);
        // }
        x[jj] = (ii / M) * SIZE2 * M + ii % M + 0 * M;
        x_1   = (ii / M) * SIZE2 * M + ii % M + 1 * M;

        // if (bx == 0 && jj == 0 && tx == 0) printf("%d %d %f\n", x[jj], x_1);

        // if (ii >= N / 2) {
        //   printf("UC: %d\n", x[jj]);
        // }


        float angle = -2 * M_PI * ((N / (M * SIZE2)) * ii - (N / SIZE2) * (ii / M)) / N;
        cuComplex weight = make_cuComplex(cos(angle), sin(angle));
        // if (bx == BIG_SET - 1 && tx == SMALL_SET - 1 && M == 1024) printf("%d %d %f\n", jj, x[jj], in[x[jj]]);
        // if (bx == 0 && tx == 0 && M == 1024) printf("%d %d %f\n", jj, x[jj], in[x[jj]]);

        y[0] =      in[x[jj]]; //shared_data[tx * coalescence + jj];
        y[1] = Mult(in[x_1], weight);

        // if (bx == BIG_SET - 1 && tx == SMALL_SET - 1 && M == 1024) printf("%d %d %f\n", jj, x[jj], in[x[jj]]);
        in[x[jj]] = Add(y[0], y[1]);
        in[x_1]   = Sub(y[0], y[1]);
      }
      cg::sync(grid);
    } 

    // for (jj = 0; jj < coalescence; jj++) {
    //   if (!valid[jj]) {
    //     printf("%d, ", (bx * coalescence + jj) * blockDim.x + tx);
    //   }
    // }
  }
}

// __global__ void Radix2_2(cuComplex* in, const unsigned int N, unsigned int M) {
//   Radix2Shift(in, N, M);

//   unsigned int k = SIZE2 * SIZE2;
//   Radix2MultShared(in, N, k);
//   for (int k = 1; k <  N; k *= SIZE2) {
//     Radix2Mult(in, N, k);
//   }
// }

__global__ void Radix4Shift(cuComplex* in,const unsigned int N, unsigned int M)
{
  for (int ii = bx * blockDim.x + tx;
       ii < N;
       ii += blockDim.x * gridDim.x) {
    unsigned int swap_idx = ii;

    unsigned int temp = 0;
    for (int jj = 0; jj < X; jj += 2) {
      temp |= (((swap_idx >> (jj + 1)) & 0x01) << (X - 1 - jj));
      temp |= (((swap_idx >> (jj + 0)) & 0x01) << (X - 2 - jj));
    }
    swap_idx = temp >> (X - M);

    if (ii < swap_idx) {
      Swap(in[swap_idx], in[ii]);
    }
  }
}

__global__ void Radix4Mult(cuComplex* in, const unsigned int N, const unsigned int M) {
  unsigned int x[SIZE4];
  cuComplex y[SIZE4];

  assert(bx * blockDim.x + tx < N / SIZE4);

  for (int ii = bx * blockDim.x + tx;
       ii < N / SIZE4;
       ii += blockDim.x * gridDim.x) {

    for (int jj = 0; jj < SIZE4; jj++) {
      x[jj] = (ii / M) * SIZE4 * M + ii % M + jj * M;
    }

    float angle = -2.0 * M_PI * (ii % M) / (SIZE4 * M);
    y[0] = in[x[0]];
    for (int jj = 1; jj < SIZE4; jj++) {
      y[jj] = Mult(in[x[jj]], make_cuComplex(cos(angle * jj), sin(angle * jj)));
    }

    in[x[0]] = Add(y[0], y[1], y[2], y[3]);
    in[x[1]] = XSubAdd(Sub(XAddSub(y[0], y[1]), y[2]), y[3]);
    in[x[2]] = Sub(Add(Sub(y[0], y[1]), y[2]), y[3]);
    in[x[3]] = XAddSub(Sub(XSubAdd(y[0], y[1]), y[2]), y[3]);
  }
}

// __global__ void Radix4(cuComplex* in, const unsigned int N, const unsigned int M) {
//   Radix4Shift(in, N, M);
//   for (int k = 1; k < N; k *= SIZE4) {
//     Radix4Mult(in, N, k);
//   }
// }


__global__ void Radix4MultShared(cuComplex* in, const unsigned int N, unsigned int k)
{
  unsigned int x[SIZE4];
  cuComplex y[SIZE4];
  __shared__ cuComplex shared_data[TILE_SIZE];

  int M = 1;
  for (int ii = bx * blockDim.x + tx;
       ii < N / SIZE4;
       ii += blockDim.x * gridDim.x) {

    for (int jj = 0; jj < SIZE4; jj++) {
      x[jj] = (ii / M) * SIZE4 * M + ii % M + jj * M;
    }

    float angle = -2.0 * M_PI * (ii % M) / (SIZE4 * M);
    y[0] = shared_data[x[0] % TILE_SIZE];
    for (int jj = 1; jj < SIZE4; jj++) {
      y[jj] = Mult(shared_data[x[jj] % TILE_SIZE], make_cuComplex(cos(angle * jj), sin(angle * jj)));
    }

    shared_data[x[0] % TILE_SIZE] = Add(y[0], y[1], y[2], y[3]);
    shared_data[x[1] % TILE_SIZE] = XSubAdd(Sub(XAddSub(y[0], y[1]), y[2]), y[3]);
    shared_data[x[2] % TILE_SIZE] = Sub(Add(Sub(y[0], y[1]), y[2]), y[3]);
    shared_data[x[3] % TILE_SIZE] = XAddSub(Sub(XSubAdd(y[0], y[1]), y[2]), y[3]);

    for (M = M * SIZE4; M < k; M *= SIZE4) {
      for (int jj = 0; jj < SIZE4; jj++) {
        x[jj] = (ii / M) * SIZE4 * M + ii % M + jj * M;
      }

      // if (bx == 1 && x[0] % TILE_SIZE == 0) {
      //   printf("%d %d %d\n", M, tx, x[0]);
      // }

      float angle = -2.0 * M_PI * (ii % M) / (SIZE4 * M);
      y[0] = in[x[0]];
      for (int jj = 1; jj < SIZE4; jj++) {
        y[jj] = Mult(in[x[jj]], make_cuComplex(cos(angle * jj), sin(angle * jj)));
      }

      shared_data[x[0] % TILE_SIZE] = Add(y[0], y[1], y[2], y[3]);
      shared_data[x[1] % TILE_SIZE] = XSubAdd(Sub(XAddSub(y[0], y[1]), y[2]), y[3]);
      shared_data[x[2] % TILE_SIZE] = Sub(Add(Sub(y[0], y[1]), y[2]), y[3]);
      shared_data[x[3] % TILE_SIZE] = XAddSub(Sub(XSubAdd(y[0], y[1]), y[2]), y[3]);
      __syncthreads();
    }

    in[x[0]] = shared_data[x[0] % TILE_SIZE];
    in[x[1]] = shared_data[x[1] % TILE_SIZE];
    in[x[2]] = shared_data[x[2] % TILE_SIZE];
    in[x[3]] = shared_data[x[3] % TILE_SIZE];
    __syncthreads();
  }
}

// __global__ void Radix4_2(cuComplex* in, const unsigned int N, const unsigned int M) {
//   Radix4Shift(in, N, M);
  
//   unsigned int k = SIZE4;
//   Radix4MultShared(in, N, k);

//   for (int k = 1; k < N; k *= SIZE4)
//     Radix4Mult(in, N, k);
// }


__global__ void Radix8Shift(cuComplex* in,const unsigned int N, unsigned int M)
{
  for (int ii = bx * blockDim.x + tx;
       ii < N;
       ii += blockDim.x * gridDim.x) {
    unsigned int swap_idx = ii;

    unsigned int temp = 0;
    for (int jj = 0; jj < X; jj++) {
      temp |= ((swap_idx >> (jj + 0)) & 0x01) << (X - 1 - jj);
    }
    swap_idx = temp >> (X - M);

    if (ii < swap_idx) {
      Swap(in[swap_idx], in[ii]);
    }

//SMALL SCALE
    // int n = log2(N);
    // unsigned int banks = N / 8;
    // unsigned int ba = ii % banks;
    // // unsigned int bi = ii * 8 / N;
    // unsigned int i = ii % 8;

    // unsigned int bi = i;
    // for (int t = 0; t < (n + 3 - 1) / 3 - 1; t++) {
    //   int sum = 0;
    //   sum += ((ba >> (n - 4 - 3 * t)) & 0x1) * pow(2, 2);
    //   sum += ((ba >> (n - 5 - 3 * t)) & 0x1) * pow(2, 1);
    //   sum += ((ba >> (n - 6 - 3 * t)) & 0x1) * pow(2, 0);
      
    //   bi += sum;
    // }
    // bi = bi % 8;
    


//CONTROVERSIAL SCALE
    // unsigned int swap_idx = ii;
    // unsigned long int temp = 0;
    // for (int jj = 0; jj < 32; jj += 3) {
    //   temp |= (((swap_idx >> (jj + 2)) & 0x01) << (X - 1 - jj));
    //   temp |= (((swap_idx >> (jj + 1)) & 0x01) << (X - 2 - jj));
    //   temp |= (((swap_idx >> (jj + 0)) & 0x01) << (X - 3 - jj));
    // }
    // swap_idx = temp >> (32 - M);

    // if (ii < swap_idx) {
    //   Swap(in[swap_idx], in[ii]);
    // }
  }
}
// This kernel performs the butterfly operations for each stage of the radix-4 FFT.
__global__ void Radix8Mult(cuComplex* in, const unsigned int N, const unsigned int M) {
//  __shared__ int shared_data[TILE_SIZE];

  unsigned int x[SIZE8];
  cuComplex y[SIZE8];

  for (int ii = bx * blockDim.x + tx;
       ii < N / SIZE8;
       ii += blockDim.x * gridDim.x) {

    for (int jj = 0; jj < SIZE8; jj++) {
      x[jj] = (ii / M) * SIZE8 * M + ii % M + jj * M;
    }

    // Compute the angle for the butterfly operations
    float angle = -2.0 * M_PI * (ii % M) / (SIZE8 * M);
    y[0] = in[x[0]];
    for (int jj = 1; jj < SIZE8; jj++) {
      y[jj] = Mult(in[x[jj]], make_cuComplex(cos(angle * jj), sin(angle * jj)));
    }

//    https://ieeexplore.ieee.org/document/6469786
//    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1328808
    const float sq2 = sqrt(2.0) / 2.0;
    in[x[0]] = make_cuComplex(y[0].x + y[1].x + y[2].x + y[3].x + y[4].x + y[5].x + y[6].x + y[7].x
                                 , y[0].y + y[1].y + y[2].y + y[3].y + y[4].y + y[5].y + y[6].y + y[7].y);


    in[x[1]] = y[0];
    in[x[1]] = Add(Mult(make_cuComplex(   0.0,   -1.0), y[1]), in[x[1]]);
    in[x[1]] = Add(Mult(make_cuComplex(  -1.0,    0.0), y[2]), in[x[1]]);
    in[x[1]] = Add(Mult(make_cuComplex(   0.0,    1.0), y[3]), in[x[1]]);
    in[x[1]] = Add(Mult(make_cuComplex(   1.0,    0.0), y[4]), in[x[1]]);
    in[x[1]] = Add(Mult(make_cuComplex(   0.0,   -1.0), y[5]), in[x[1]]);
    in[x[1]] = Add(Mult(make_cuComplex(  -1.0,    0.0), y[6]), in[x[1]]);
    in[x[1]] = Add(Mult(make_cuComplex(   0.0,    1.0), y[7]), in[x[1]]);

    in[x[2]] = y[0];
    in[x[2]] = Add(Mult(make_cuComplex( -1.0,  0.0), y[1]), in[x[2]]);
    in[x[2]] = Add(Mult(make_cuComplex(  1.0,  0.0), y[2]), in[x[2]]);
    in[x[2]] = Add(Mult(make_cuComplex( -1.0,  0.0), y[3]), in[x[2]]);
    in[x[2]] = Add(Mult(make_cuComplex(  1.0,  0.0), y[4]), in[x[2]]);
    in[x[2]] = Add(Mult(make_cuComplex( -1.0,  0.0), y[5]), in[x[2]]);
    in[x[2]] = Add(Mult(make_cuComplex(  1.0,  0.0), y[6]), in[x[2]]);
    in[x[2]] = Add(Mult(make_cuComplex( -1.0,  0.0), y[7]), in[x[2]]);

    in[x[3]] = y[0];
    in[x[3]] = Add(Mult(make_cuComplex(   0.0,  1.0), y[1]), in[x[3]]);
    in[x[3]] = Add(Mult(make_cuComplex(  -1.0,  0.0), y[2]), in[x[3]]);
    in[x[3]] = Add(Mult(make_cuComplex(   0.0, -1.0), y[3]), in[x[3]]);
    in[x[3]] = Add(Mult(make_cuComplex(   1.0,  0.0), y[4]), in[x[3]]);
    in[x[3]] = Add(Mult(make_cuComplex(   0.0,  1.0), y[5]), in[x[3]]);
    in[x[3]] = Add(Mult(make_cuComplex(  -1.0,  0.0), y[6]), in[x[3]]);
    in[x[3]] = Add(Mult(make_cuComplex(   0.0, -1.0), y[7]), in[x[3]]);

    in[x[4]] = y[0];
    in[x[4]] = Add(Mult(make_cuComplex(sq2, -sq2), y[1]), in[x[4]]);
    in[x[4]] = Add(Mult(make_cuComplex( 0.0, -1.0), y[2]), in[x[4]]);
    in[x[4]] = Add(Mult(make_cuComplex(-sq2, -sq2), y[3]), in[x[4]]);
    in[x[4]] = Add(Mult(make_cuComplex( -1.0, 0.0), y[4]), in[x[4]]);
    in[x[4]] = Add(Mult(make_cuComplex(-sq2, sq2), y[5]), in[x[4]]);
    in[x[4]] = Add(Mult(make_cuComplex( 0.0, 1.0), y[6]), in[x[4]]);
    in[x[4]] = Add(Mult(make_cuComplex(sq2, sq2), y[7]), in[x[4]]);

    in[x[5]] = y[0];
    in[x[5]] = Add(Mult(make_cuComplex(  -sq2,   -sq2), y[1]), in[x[5]]);
    in[x[5]] = Add(Mult(make_cuComplex(   0.0,  1.0), y[2]), in[x[5]]);
    in[x[5]] = Add(Mult(make_cuComplex(   sq2,   -sq2), y[3]), in[x[5]]);
    in[x[5]] = Add(Mult(make_cuComplex(  -1.0,   0.0), y[4]), in[x[5]]);
    in[x[5]] = Add(Mult(make_cuComplex(   sq2,  sq2), y[5]), in[x[5]]);
    in[x[5]] = Add(Mult(make_cuComplex(   0.0,   -1.0), y[6]), in[x[5]]);
    in[x[5]] = Add(Mult(make_cuComplex(  -sq2,  sq2), y[7]), in[x[5]]);

    in[x[6]] = y[0];
    in[x[6]] = Add(Mult(make_cuComplex(  -sq2,  sq2), y[1]), in[x[6]]);
    in[x[6]] = Add(Mult(make_cuComplex( 0.0,  -1.0), y[2]), in[x[6]]);
    in[x[6]] = Add(Mult(make_cuComplex(  sq2, sq2), y[3]), in[x[6]]);
    in[x[6]] = Add(Mult(make_cuComplex(  -1.0,  0.0), y[4]), in[x[6]]);
    in[x[6]] = Add(Mult(make_cuComplex(  sq2,  -sq2), y[5]), in[x[6]]);
    in[x[6]] = Add(Mult(make_cuComplex( 0.0,  1.0), y[6]), in[x[6]]);
    in[x[6]] = Add(Mult(make_cuComplex(  -sq2, -sq2), y[7]), in[x[6]]);

    in[x[7]] = y[0];
    in[x[7]] = Add(Mult(make_cuComplex(   sq2,   sq2), y[1]), in[x[7]]);
    in[x[7]] = Add(Mult(make_cuComplex(   0.0,   1.0), y[2]), in[x[7]]);
    in[x[7]] = Add(Mult(make_cuComplex(  -sq2,   sq2), y[3]), in[x[7]]);
    in[x[7]] = Add(Mult(make_cuComplex(  -1.0,   0.0), y[4]), in[x[7]]);
    in[x[7]] = Add(Mult(make_cuComplex(  -sq2,  -sq2), y[5]), in[x[7]]);
    in[x[7]] = Add(Mult(make_cuComplex(   0.0,  -1.0), y[6]), in[x[7]]);
    in[x[7]] = Add(Mult(make_cuComplex(   sq2,  -sq2), y[7]), in[x[7]]);

/*
    in[x[1]] = y[0];
    in[x[1]] = Add(Mult(make_cuComplex(cos(2 * PI * 1 / 8), sin(2 * PI * 1 / 8)), y[1]), in[x[1]]);
    in[x[1]] = Add(Mult(make_cuComplex(cos(2 * PI * 2 / 8), sin(2 * PI * 2 / 8)), y[2]), in[x[1]]);
    in[x[1]] = Add(Mult(make_cuComplex(cos(2 * PI * 3 / 8), sin(2 * PI * 3 / 8)), y[3]), in[x[1]]);
    in[x[1]] = Add(Mult(make_cuComplex(cos(2 * PI * 4 / 8), sin(2 * PI * 4 / 8)), y[4]), in[x[1]]);
    in[x[1]] = Add(Mult(make_cuComplex(cos(2 * PI * 5 / 8), sin(2 * PI * 5 / 8)), y[5]), in[x[1]]);
    in[x[1]] = Add(Mult(make_cuComplex(cos(2 * PI * 6 / 8), sin(2 * PI * 6 / 8)), y[6]), in[x[1]]);
    in[x[1]] = Add(Mult(make_cuComplex(cos(2 * PI * 7 / 8), sin(2 * PI * 7 / 8)), y[7]), in[x[1]]);

    in[x[2]] = y[0];
    in[x[2]] = Add(Mult(make_cuComplex(cos(2 * PI * 2 / 8), sin(2 * PI * 2 / 8)), y[1]), in[x[2]]);
    in[x[2]] = Add(Mult(make_cuComplex(cos(2 * PI * 4 / 8), sin(2 * PI * 4 / 8)), y[2]), in[x[2]]);
    in[x[2]] = Add(Mult(make_cuComplex(cos(2 * PI * 6 / 8), sin(2 * PI * 6 / 8)), y[3]), in[x[2]]);
    in[x[2]] = Add(Mult(make_cuComplex(cos(2 * PI * 0 / 8), sin(2 * PI * 0 / 8)), y[4]), in[x[2]]);
    in[x[2]] = Add(Mult(make_cuComplex(cos(2 * PI * 2 / 8), sin(2 * PI * 2 / 8)), y[5]), in[x[2]]);
    in[x[2]] = Add(Mult(make_cuComplex(cos(2 * PI * 4 / 8), sin(2 * PI * 4 / 8)), y[6]), in[x[2]]);
    in[x[2]] = Add(Mult(make_cuComplex(cos(2 * PI * 6 / 8), sin(2 * PI * 6 / 8)), y[7]), in[x[2]]);

    in[x[3]] = y[0];
    in[x[3]] = Add(Mult(make_cuComplex(cos(2 * PI * 3 / 8), sin(2 * PI * 3 / 8)), y[1]), in[x[3]]);
    in[x[3]] = Add(Mult(make_cuComplex(cos(2 * PI * 6 / 8), sin(2 * PI * 6 / 8)), y[2]), in[x[3]]);
    in[x[3]] = Add(Mult(make_cuComplex(cos(2 * PI * 9 / 8), sin(2 * PI * 9 / 8)), y[3]), in[x[3]]);
    in[x[3]] = Add(Mult(make_cuComplex(cos(2 * PI * 4 / 8), sin(2 * PI * 4 / 8)), y[4]), in[x[3]]);
    in[x[3]] = Add(Mult(make_cuComplex(cos(2 * PI * 7 / 8), sin(2 * PI * 7 / 8)), y[5]), in[x[3]]);
    in[x[3]] = Add(Mult(make_cuComplex(cos(2 * PI * 2 / 8), sin(2 * PI * 2 / 8)), y[6]), in[x[3]]);
    in[x[3]] = Add(Mult(make_cuComplex(cos(2 * PI * 6 / 8), sin(2 * PI * 6 / 8)), y[7]), in[x[3]]);

    in[x[4]] = y[0];
    in[x[4]] = Add(Mult(make_cuComplex(cos(2 * PI * 4 / 8), sin(2 * PI * 4 / 8)), y[1]), in[x[4]]);
    in[x[4]] = Add(Mult(make_cuComplex(cos(2 * PI * 0 / 8), sin(2 * PI * 0 / 8)), y[2]), in[x[4]]);
    in[x[4]] = Add(Mult(make_cuComplex(cos(2 * PI * 4 / 8), sin(2 * PI * 4 / 8)), y[3]), in[x[4]]);
    in[x[4]] = Add(Mult(make_cuComplex(cos(2 * PI * 0 / 8), sin(2 * PI * 0 / 8)), y[4]), in[x[4]]);
    in[x[4]] = Add(Mult(make_cuComplex(cos(2 * PI * 4 / 8), sin(2 * PI * 4 / 8)), y[5]), in[x[4]]);
    in[x[4]] = Add(Mult(make_cuComplex(cos(2 * PI * 0 / 8), sin(2 * PI * 0 / 8)), y[6]), in[x[4]]);
    in[x[4]] = Add(Mult(make_cuComplex(cos(2 * PI * 4 / 8), sin(2 * PI * 4 / 8)), y[7]), in[x[4]]);

    in[x[5]] = y[0];
    in[x[5]] = Add(Mult(make_cuComplex(cos(2 * PI * 5 / 8), sin(2 * PI * 5 / 8)), y[1]), in[x[5]]);
    in[x[5]] = Add(Mult(make_cuComplex(cos(2 * PI * 2 / 8), sin(2 * PI * 2 / 8)), y[2]), in[x[5]]);
    in[x[5]] = Add(Mult(make_cuComplex(cos(2 * PI * 7 / 8), sin(2 * PI * 7 / 8)), y[3]), in[x[5]]);
    in[x[5]] = Add(Mult(make_cuComplex(cos(2 * PI * 4 / 8), sin(2 * PI * 4 / 8)), y[4]), in[x[5]]);
    in[x[5]] = Add(Mult(make_cuComplex(cos(2 * PI * 1 / 8), sin(2 * PI * 1 / 8)), y[5]), in[x[5]]);
    in[x[5]] = Add(Mult(make_cuComplex(cos(2 * PI * 6 / 8), sin(2 * PI * 6 / 8)), y[6]), in[x[5]]);
    in[x[5]] = Add(Mult(make_cuComplex(cos(2 * PI * 3 / 8), sin(2 * PI * 3 / 8)), y[7]), in[x[5]]);

    in[x[6]] = y[0];
    in[x[6]] = Add(Mult(make_cuComplex(cos(2 * PI * 6 / 8), sin(2 * PI * 6 / 8)), y[1]), in[x[6]]);
    in[x[6]] = Add(Mult(make_cuComplex(cos(2 * PI * 4 / 8), sin(2 * PI * 4 / 8)), y[2]), in[x[6]]);
    in[x[6]] = Add(Mult(make_cuComplex(cos(2 * PI * 2 / 8), sin(2 * PI * 2 / 8)), y[3]), in[x[6]]);
    in[x[6]] = Add(Mult(make_cuComplex(cos(2 * PI * 0 / 8), sin(2 * PI * 0 / 8)), y[4]), in[x[6]]);
    in[x[6]] = Add(Mult(make_cuComplex(cos(2 * PI * 6 / 8), sin(2 * PI * 6 / 8)), y[5]), in[x[6]]);
    in[x[6]] = Add(Mult(make_cuComplex(cos(2 * PI * 4 / 8), sin(2 * PI * 4 / 8)), y[6]), in[x[6]]);
    in[x[6]] = Add(Mult(make_cuComplex(cos(2 * PI * 2 / 8), sin(2 * PI * 2 / 8)), y[7]), in[x[6]]);

    in[x[7]] = y[0];
    in[x[7]] = Add(Mult(make_cuComplex(cos(2 * PI * 7 / 8), sin(2 * PI * 7 / 8)), y[1]), in[x[7]]);
    in[x[7]] = Add(Mult(make_cuComplex(cos(2 * PI * 6 / 8), sin(2 * PI * 6 / 8)), y[2]), in[x[7]]);
    in[x[7]] = Add(Mult(make_cuComplex(cos(2 * PI * 5 / 8), sin(2 * PI * 5 / 8)), y[3]), in[x[7]]);
    in[x[7]] = Add(Mult(make_cuComplex(cos(2 * PI * 4 / 8), sin(2 * PI * 4 / 8)), y[4]), in[x[7]]);
    in[x[7]] = Add(Mult(make_cuComplex(cos(2 * PI * 3 / 8), sin(2 * PI * 3 / 8)), y[5]), in[x[7]]);
    in[x[7]] = Add(Mult(make_cuComplex(cos(2 * PI * 2 / 8), sin(2 * PI * 2 / 8)), y[6]), in[x[7]]);
    in[x[7]] = Add(Mult(make_cuComplex(cos(2 * PI * 1 / 8), sin(2 * PI * 1 / 8)), y[7]), in[x[7]]);
*/
  }
}

// __global__ void Radix8(cuComplex* in, const unsigned int N, const unsigned int M) {
//   Radix8Shift(in, N, M);
//   for (int k = 1; k < N; k *= SIZE8)
//     Radix8Mult(in, N, k);
// }

#endif //RADIX2KERNEL_CUH

/*

    in[x[1]] = y[0];
    in[x[1]] = Add(Mult(make_cuComplex(   0.0,   -1.0), y[1]), in[x[1]]);
    in[x[1]] = Add(Mult(make_cuComplex(  -1.0,    0.0), y[2]), in[x[1]]);
    in[x[1]] = Add(Mult(make_cuComplex(   0.0,    1.0), y[3]), in[x[1]]);
    in[x[1]] = Add(Mult(make_cuComplex(   1.0,    0.0), y[4]), in[x[1]]);
    in[x[1]] = Add(Mult(make_cuComplex(   0.0,   -1.0), y[5]), in[x[1]]);
    in[x[1]] = Add(Mult(make_cuComplex(  -1.0,    0.0), y[6]), in[x[1]]);
    in[x[1]] = Add(Mult(make_cuComplex(   0.0,    1.0), y[7]), in[x[1]]);

    in[x[2]] = y[0];
    in[x[2]] = Add(Mult(make_cuComplex( -1.0,  0.0), y[1]), in[x[2]]);
    in[x[2]] = Add(Mult(make_cuComplex(  1.0,  0.0), y[2]), in[x[2]]);
    in[x[2]] = Add(Mult(make_cuComplex( -1.0,  0.0), y[3]), in[x[2]]);
    in[x[2]] = Add(Mult(make_cuComplex(  1.0,  0.0), y[4]), in[x[2]]);
    in[x[2]] = Add(Mult(make_cuComplex( -1.0,  0.0), y[5]), in[x[2]]);
    in[x[2]] = Add(Mult(make_cuComplex(  1.0,  0.0), y[6]), in[x[2]]);
    in[x[2]] = Add(Mult(make_cuComplex( -1.0,  0.0), y[7]), in[x[2]]);

    in[x[3]] = y[0];
    in[x[3]] = Add(Mult(make_cuComplex(   0.0,  1.0), y[1]), in[x[3]]);
    in[x[3]] = Add(Mult(make_cuComplex(  -1.0,  0.0), y[2]), in[x[3]]);
    in[x[3]] = Add(Mult(make_cuComplex(   0.0, -1.0), y[3]), in[x[3]]);
    in[x[3]] = Add(Mult(make_cuComplex(   1.0,  0.0), y[4]), in[x[3]]);
    in[x[3]] = Add(Mult(make_cuComplex(   0.0,  1.0), y[5]), in[x[3]]);
    in[x[3]] = Add(Mult(make_cuComplex(  -1.0,  0.0), y[6]), in[x[3]]);
    in[x[3]] = Add(Mult(make_cuComplex(   0.0, -1.0), y[7]), in[x[3]]);

    in[x[4]] = y[0];
    in[x[4]] = Add(Mult(make_cuComplex(sq2, -sq2), y[1]), in[x[4]]);
    in[x[4]] = Add(Mult(make_cuComplex( 0.0, -1.0), y[2]), in[x[4]]);
    in[x[4]] = Add(Mult(make_cuComplex(-sq2, -sq2), y[3]), in[x[4]]);
    in[x[4]] = Add(Mult(make_cuComplex( -1.0, 0.0), y[4]), in[x[4]]);
    in[x[4]] = Add(Mult(make_cuComplex(-sq2, sq2), y[5]), in[x[4]]);
    in[x[4]] = Add(Mult(make_cuComplex( 0.0, 1.0), y[6]), in[x[4]]);
    in[x[4]] = Add(Mult(make_cuComplex(sq2, sq2), y[7]), in[x[4]]);

    in[x[5]] = y[0];
    in[x[5]] = Add(Mult(make_cuComplex(  -sq2,   -sq2), y[1]), in[x[5]]);
    in[x[5]] = Add(Mult(make_cuComplex(   0.0,  1.0), y[2]), in[x[5]]);
    in[x[5]] = Add(Mult(make_cuComplex(   sq2,   -sq2), y[3]), in[x[5]]);
    in[x[5]] = Add(Mult(make_cuComplex(  -1.0,   0.0), y[4]), in[x[5]]);
    in[x[5]] = Add(Mult(make_cuComplex(   sq2,  sq2), y[5]), in[x[5]]);
    in[x[5]] = Add(Mult(make_cuComplex(   0.0,   -1.0), y[6]), in[x[5]]);
    in[x[5]] = Add(Mult(make_cuComplex(  -sq2,  sq2), y[7]), in[x[5]]);

    in[x[6]] = y[0];
    in[x[6]] = Add(Mult(make_cuComplex(  -sq2,  sq2), y[1]), in[x[6]]);
    in[x[6]] = Add(Mult(make_cuComplex( 0.0,  -1.0), y[2]), in[x[6]]);
    in[x[6]] = Add(Mult(make_cuComplex(  sq2, sq2), y[3]), in[x[6]]);
    in[x[6]] = Add(Mult(make_cuComplex(  -1.0,  0.0), y[4]), in[x[6]]);
    in[x[6]] = Add(Mult(make_cuComplex(  sq2,  -sq2), y[5]), in[x[6]]);
    in[x[6]] = Add(Mult(make_cuComplex( 0.0,  1.0), y[6]), in[x[6]]);
    in[x[6]] = Add(Mult(make_cuComplex(  -sq2, -sq2), y[7]), in[x[6]]);

    in[x[7]] = y[0];
    in[x[7]] = Add(Mult(make_cuComplex(   sq2,   sq2), y[1]), in[x[7]]);
    in[x[7]] = Add(Mult(make_cuComplex(   0.0,   1.0), y[2]), in[x[7]]);
    in[x[7]] = Add(Mult(make_cuComplex(  -sq2,   sq2), y[3]), in[x[7]]);
    in[x[7]] = Add(Mult(make_cuComplex(  -1.0,   0.0), y[4]), in[x[7]]);
    in[x[7]] = Add(Mult(make_cuComplex(  -sq2,  -sq2), y[5]), in[x[7]]);
    in[x[7]] = Add(Mult(make_cuComplex(   0.0,  -1.0), y[6]), in[x[7]]);
    in[x[7]] = Add(Mult(make_cuComplex(   sq2,  -sq2), y[7]), in[x[7]]);

*/