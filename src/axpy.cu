// Project Header
#include "axpy.hpp"

// Project Kernel
#include "axpy.cuh"
#include "gpuErrorChk.h"

// Standard Headers
#include <iostream>

#define THREADSPERBLOCK 1024

// Constructors
template<typename T>
Axpy<T>::Axpy(int N, int deviceID ) : N_(N), device_id_(deviceID) {
  int device_Count;
  cudaGetDeviceCount(&device_Count);
  printf("\n\nNumber of GPUs: %d\n", device_Count);
  // Device Selection
  cudaDeviceProp deviceProp;
  int status = cudaSetDevice(device_id_);
  cudaGetDevice(&device_id_);   	
  cudaGetDeviceProperties(&deviceProp, device_id_);
  printf("Using device %d: %s\n", device_id_, deviceProp.name);



	gpuErrchk( cudaMalloc((void**)&d_x_, N_ * sizeof(T)) );
	gpuErrchk( cudaMalloc((void**)&d_y_, N_ * sizeof(T)) );
};  // Default constructor

// Destructor
template<typename T>
Axpy<T>::~Axpy() {  
	gpuErrchk( cudaFree(d_x_) );
	gpuErrchk( cudaFree(d_y_) );
};

template<typename T>
void Axpy<T>::Load(T * h_x, T * h_y) {  
	// Copy input vectors from host to device
	gpuErrchk( cudaMemcpy(d_x_, h_x, N_ * sizeof(T), cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(d_y_, h_y, N_ * sizeof(T), cudaMemcpyHostToDevice) );
};

template<typename T>
void Axpy<T>::Execute(T a) {  
  dim3 threads_per_block(THREADSPERBLOCK);
  dim3 blocks_per_grid((N_ + THREADSPERBLOCK - 1) / THREADSPERBLOCK);
  
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
  cudaEventRecord(start);
  saxpy<<<blocks_per_grid, threads_per_block>>>(a, d_x_, d_y_, N_);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  
  printf("Operating Time(millsec): %f\n", milliseconds);
  
	// Wait for kernel to finish
	gpuErrchk( cudaGetLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
};

template<typename T>
void Axpy<T>::Purge(T * h_y) {  
	// Copy output vector from device to host
	gpuErrchk( cudaMemcpy(h_y, d_y_, N_ * sizeof(T), cudaMemcpyDeviceToHost) );
};

template class Axpy<float>;
template class Axpy<double>;


