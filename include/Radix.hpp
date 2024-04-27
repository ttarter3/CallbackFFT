#ifndef RADIX2_HPP
#define RADIX2_HPP

#include <complex>
#include <cuComplex.h>
#include <cufft.h>

template<typename T>
class Radix {
private:
    // Private member variables and methods
    cuComplex * d_x_;
    int N_;
    int N_populated_;
    int M_;
    int device_id_;

    cufftHandle plan_;

    void GPUKernel(cuComplex* x, const unsigned int N, const unsigned int M, int rad = 2);

public:
    // Constructors
    Radix( int N_populated, int N, int deviceID=-1 );

    // Destructor
    ~Radix();

    void Load(std::complex<T> * h_x);
    void Execute(int radix);
    void Purge(std::complex<T> * h_x);


};

#endif // RADIX2_HPP
