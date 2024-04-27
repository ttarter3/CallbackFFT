// Project Header
#include "Radix.hpp"

// Project Kernel
#include "RadixKernel.cuh"
#include "gpuErrorChk.h"
#include "Define.h"

// Standard Headers
#include <iostream>
#include <cstdio>
#include <complex>



template<typename T>
void Radix<T>::GPUKernel(cuComplex* in, const unsigned int N, const unsigned int M, int alg) {
  int rad = 0;
  switch (alg) {
        case 1:
          rad = 2;
          if (abs(remainderl(logl(powl(2, M)), logl(2.0))) < 1.0E-5) {
            // Radix2<<<dim3((N + TPB_512 - 1) / TPB_512, 1, 1), TPB_512 >>>(in, N, M);
            Radix2Shift<<<dim3((N + TPB_512 - 1) / TPB_512, 1, 1), dim3(TPB_512,1,1) >>>(in, N, M);
            for (int k = 1; k <  N; k *= SIZE2) {
              Radix2Mult<<<dim3((N + TPB_512 - 1) / TPB_512, 1, 1), dim3(TPB_512,1,1) >>>(in, N, k);
            }
          } else {
            std::cout << "Error: Radix "<< rad << " must statify 0 == " << remainder(log(pow(2, M)), log(2.0)) << std::endl;
          }
        break;            // and exits the switch
        case 2:
          rad = 4;
          if (abs(remainderl(logl(powl(2, M)), logl(4.0))) < 1.0E-5) {
            // Radix4<<<dim3((N + TPB_512 - 1) / TPB_512, 1, 1), TPB_512 >>>(in, N, M);
          Radix4Shift<<<dim3((N + TPB_512 - 1) / TPB_512, 1, 1), dim3(TPB_512,1,1) >>>(in, N, M);
          for (int k = 1; k < N; k *= SIZE4) {
            Radix4Mult<<<dim3((N + TPB_512 - 1) / TPB_512, 1, 1), dim3(TPB_512,1,1) >>>(in, N, k);
          }
          } else {
            std::cout << "Error: Radix "<< rad << " must statify 0 == " << abs(remainderl(logl(powl(2, M)), logl(4.0))) << std::endl;
          }
        break;
        case 3:
          rad = 8;
          if (abs(remainderl(logl(powl(2, M)), logl(8.0))) < 1.0E-5) {
            // Radix8<<< dim3((N + TPB_512 - 1) / TPB_512, 1, 1), TPB_512 >>>(in, N, M);
            Radix8Shift<<< dim3((N + TPB_512 - 1) / TPB_512, 1, 1), dim3(TPB_512,1,1) >>>(in, N, M);
            for (int k = 1; k < N; k *= SIZE8)
              Radix8Mult<<< dim3((N + TPB_512 - 1) / TPB_512, 1, 1), dim3(TPB_512,1,1) >>>(in, N, k);

          } else {
            std::cout << "Error: Radix "<< rad << " must statify 0 == " << remainder(log(pow(2, M)), log(8.0)) << std::endl;
          }
        break;
        case 4:
          rad = 2;
          if (abs(remainderl(logl(powl(2, M)), logl(2.0))) < 1.0E-5 && M > 9) {
            // Radix2_2<<< dim3((N + TPB_512 - 1) / TPB_512, 1, 1), TPB_512 >>>(in, N, M);
            Radix2Shift<<< dim3((N + TPB_512 - 1) / TPB_512, 1, 1), TPB_512 >>>(in, N, M);

            unsigned int k = SIZE2 * SIZE2;
            Radix2MultShared<<< dim3((N + TPB_512 - 1) / TPB_512, 1, 1), dim3(TPB_512, 1, 1) >>>(in, N, k);
            for (int k = 1; k <  N; k *= SIZE2) {
              Radix2Mult<<< dim3((N + TPB_512 - 1) / TPB_512, 1, 1), dim3(TPB_512, 1, 1) >>>(in, N, k);
            }
          } else {
            std::cout << "Error: Radix "<< rad << " must statify 0 == " << remainder(log(pow(2, M)), log(2.0)) << std::endl;
          }
        break;
        case 5:
          rad = 2;
          if (abs(remainderl(logl(powl(2, M)), logl(2.0))) < 1.0E-5) {
            // Radix2<<<dim3((N + TPB_512 - 1) / TPB_512, 1, 1), TPB_512 >>>(in, N, M);
            Radix2Shift<<<dim3((N + TPB_512 - 1) / TPB_512, 1, 1), dim3(TPB_512, 1, 1) >>>(in, N, M);
            for (int k = 1; k <  N; k *= SIZE2) {
              Radix2Mult2XThread<<<dim3((N + TPB_512 - 1) / TPB_512 * 2, 1, 1), dim3(TPB_512, 1, 1) >>>(in, N, k);
            }
          } else {
            std::cout << "Error: Radix "<< rad << " must statify 0 == " << remainder(log(pow(2, M)), log(2.0)) << std::endl;
          }
          break;
        case 6:
          rad = 2;
          if (abs(remainderl(logl(powl(2, M)), logl(2.0))) < 1.0E-5) {
            // Radix2<<<dim3((N + TPB_512 - 1) / TPB_512, 1, 1), TPB_512 >>>(in, N, M);
            Radix2Shift<<<dim3((N + TPB_512 - 1) / TPB_512, 1, 1), TPB_512 >>>(in, N, M);
            int k = 1;
            Radix2Mult1st<<<dim3((N + TPB_512 - 1) / TPB_512 * 2, 1, 1), dim3(TPB_512, 1, 1) >>>(in, N, k);
            k = 2 * TPB_512;

            for (k = k; k < N; k *= SIZE2) {
              Radix2Mult<<< dim3((N + TPB_512 - 1) / TPB_512, 1, 1), dim3(TPB_512, 1, 1) >>>(in, N, k);
            }
          } else {
            std::cout << "Error: Radix "<< rad << " must statify 0 == " << remainder(log(pow(2, M)), log(2.0)) << std::endl;
          }
          break;
        case 7:
          rad = 2;
          if (abs(remainderl(logl(powl(2, M)), logl(2.0))) < 1.0E-5) {
            // Radix2<<<dim3((N + TPB_512 - 1) / TPB_512, 1, 1), TPB_512 >>>(in, N, M);
            Radix2Shift<<<dim3((N + TPB_512 - 1) / TPB_512, 1, 1), TPB_512 >>>(in, N, M);
            int k = 1;
            Radix2Mult1st<<<dim3((N + TPB_512 - 1) / TPB_512 * 2, 1, 1), dim3(TPB_512, 1, 1) >>>(in, N, k);
            k = 2 * TPB_512;

            Radix2Mult2nd<<<dim3((N + TPB_512 - 1) / TPB_512 * 2, 1, 1), dim3(TPB_512, 1, 1) >>>(in, N, k);

            for (k = k; k < N; k *= SIZE2) {
              Radix2Mult<<< dim3((N + TPB_512 - 1) / TPB_512, 1, 1), dim3(TPB_512, 1, 1) >>>(in, N, k);
            }
          } else {
            std::cout << "Error: Radix "<< rad << " must statify 0 == " << remainder(log(pow(2, M)), log(2.0)) << std::endl;
          }
          break;
        
        default:
          std::cout << "Using Cuda FFT" << std::endl;
          cufftExecC2C(plan_, in, in, CUFFT_FORWARD);
          return;
  }

  std::cout << "Radix " << rad << ":" << std::endl;
}

// Constructors
template<typename T>
Radix<T>::Radix( int N_populated, int N, int deviceID) :
  N_populated_(N_populated)
  , N_(N)
  , M_(log2((float)N_))
  , device_id_(deviceID) {
  std::cout << __FUNCTION__ << std::endl;

  int device_Count = -1;

  printf("Number of GPUs: %d\n", device_Count);
  cudaGetDeviceCount(&device_Count);

  if (device_id_ != -1) {
    // Device Selection
    int status = cudaSetDevice(device_id_);
  }

  cudaDeviceProp deviceProp;
  cudaGetDevice(&device_id_);   	
  cudaGetDeviceProperties(&deviceProp, device_id_);
  printf("Using device %d: %s \n", device_id_, deviceProp.name);

  printf("Device %d: %s\n", device_id_, deviceProp.name);
  printf("Max shared memory per block: %lu bytes\n", deviceProp.sharedMemPerBlock);

  gpuErrchk( cudaMalloc((void**)&d_x_, N_ * sizeof(std::complex<T>)) );

  cufftPlan1d(&plan_, N * BATCH, CUFFT_C2C, BATCH);
};  // Default constructor

// Destructor
template<typename T>
Radix<T>::~Radix() {
  std::cout << __FUNCTION__ << std::endl;
  std::cout << std::endl;

  gpuErrchk( cudaFree(d_x_) );
  cufftDestroy(plan_);
};

template<typename T>
void Radix<T>::Load(std::complex<T> * h_x) {
  std::cout << __FUNCTION__ << std::endl;
	// Copy input vectors from host to device
	gpuErrchk( cudaMemcpy(d_x_, h_x, N_ * sizeof(std::complex<T>), cudaMemcpyHostToDevice) );
};

template<typename T>
void Radix<T>::Execute(int alg) {
  std::cout << __FUNCTION__ << std::endl;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
  cudaEventRecord(start);

  GPUKernel(d_x_, N_, M_, alg);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  
  printf("Alg %d: Operating Time(millsec): %f\n", alg, milliseconds);
  
	// Wait for kernel to finish
	gpuErrchk( cudaGetLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
};

template<typename T>
void Radix<T>::Purge(std::complex<T> * h_x) {
  std::cout << __FUNCTION__ << std::endl;
	// Copy output vector from device to host
	gpuErrchk( cudaMemcpy(h_x, d_x_, N_ * sizeof(std::complex<T>), cudaMemcpyDeviceToHost) );
};

template class Radix<float>;
template class Radix<double>;