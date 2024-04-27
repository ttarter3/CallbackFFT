
#ifndef INCLUDE_CUFFTFFT_HPP
#define INCLUDE_CUFFTFFT_HPP

// Standard Headers
#include <cufft.h>
#include <vector>


template <typename T>
class CuFFTFFT {
public:
  CuFFTFFT(int z_pad_2_size, int batch);
  ~CuFFTFFT();

  void Load(std::vector<T> & input_array);
  unsigned int Execute(int operation);
  void Purge(std::vector<T> & output_array);
private:
  int d_size_;
  int z_pad_2_size_;
  int batch_;

  cufftHandle plan_;
  cufftResult result_;

  T* d_in_data_;
  T* d_out_data_;
};

#endif // INCLUDE_CUFFTFFT_HPP
