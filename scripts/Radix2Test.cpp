//
// Created by ttarter on 3/20/24.
//

#include <Radix.hpp>
#include "findIndexOfMax.h"
#include "Vector2File.h"
#include "FileTools.hpp"

// Standard Header
#include <iostream>
#include <vector>
#include <cmath>
#include <complex>
#include <omp.h>
#include <cstdlib>
#include <iomanip>

#define TYPE float

// Function to perform FFT shift
template<typename T>
void FFTShift(std::vector<std::complex<T>>& data) {
  size_t N = data.size();

  // Calculate the center index
  size_t center = N / 2;

  // Perform the swap
  for (size_t i = 0; i < center; ++i) {
    std::swap(data[i], data[i + center]);
  }
}


template<typename T>
void generateSignal(std::vector<std::complex<T>>& x, std::vector<T> f) {
  T fs = f[0];

  std::vector<float> mag(f.size(), 0); for (int ii = 0; ii < mag.size();ii++) { mag[ii] = 15; }//rand() % 50 + 10; }

  #pragma omp parallel for
  for (int ii = 0; ii < x.size(); ii++) {
    auto t = 1.0 / fs * ii;

    #pragma omp critical
    {
      double sinValue = 0;
      for (int jj = 1; jj < f.size(); jj++) {
        sinValue += mag[jj] * std::sin(2 * M_PI * f[jj] * t); // Calculate sin value
      }
      x[ii].real(sinValue); // Set real part of complex number
      x[ii].imag(0.0);      // Set imaginary part of complex number
    }
  }
}

int main(int argc, char* argv[]) {
    int device, usable_bytes, n, m, two_pow_m;

    std::cout << "start" << std::endl;
    // std::this_thread::sleep_for(std::chrono::nanoseconds(10));
    // std::this_thread::sleep_until(std::chrono::system_clock::now() + std::chrono::seconds(1));

    std::vector<TYPE> f({ 16.0f, 1.0f, 2.0f, 4.0f });

    device = -1; // NVIDIA A2

    usable_bytes = 2.0E8; // n: 25000000	m: 25	2^M:33554432
    usable_bytes = 1.0E8; // n: 12500000	m: 24	2^M:16777216

    n = usable_bytes / sizeof(std::complex<TYPE>);
    m = log2((float)n) + 1;

    int iteration = 1;
    if (argc > 1) {
      iteration = std::atoi(argv[1]);
    }
    std::vector<int> alg_2_run({0, 1, 2, 6});

    if (argc > 2) {
      alg_2_run.clear();
      for (int ii = 2; ii < argc; ii++) {
        alg_2_run.push_back(std::atoi(argv[ii]));
      }
    }

    const char* folderPath = "./data";
    if (createFolder(folderPath)) {
        std::cout << "Folder handling complete." << std::endl;
    } else {
        std::cerr << "Folder handling failed." << std::endl;
    }

    for (m = 12; m < 25; m+=2) {
      two_pow_m = pow(2, m); // x: 1050000000
      std::cout << "n: " << n << "\tm: " << m << "\t2^M:" << two_pow_m << std::endl;
      std::vector<std::complex<TYPE>> solution(0);
      for (int ii_alg = 0; ii_alg < alg_2_run.size(); ii_alg++) {
        for (int jj = 0; jj < iteration; jj++) {
          int alg = alg_2_run[ii_alg];

          std::vector<std::complex<TYPE>> x(two_pow_m);
          generateSignal(x, f);
          try {
          Radix<TYPE> radix(n, x.size(), device);
          radix.Load(x.data());
          radix.Execute(alg);
          radix.Purge(x.data());

          if (alg == 0) {
            solution.resize(x.size());
            std::copy(x.begin(), x.end(), solution.begin());
          }
          if (solution.size() > 0) {
            if (solution.size() != x.size()) {
              std::cout << "Error: Invalid Vector Size" << std::endl;
            }

            double error = 0;
            for (int ii = 0; ii < solution.size(); ii++) {
              error += std::pow(std::abs(solution[ii] - x[ii]), 2);
            }; error = std::sqrt(error);
            if (error > 1.0E-3) {
              std::cout << "Error: Measurment Error Greater than expected(" <<  std::setprecision(12) << error << ")" << std::endl;
            }
          }

          // std::string config_file = std::string(folderPath) + "/Config." + std::to_string(alg) + ".bin";
          // Vector2File(f, config_file);

          // FFTShift(x);

          // std::string freq_data_file = std::string(folderPath) + "/FreqData." + std::to_string(alg) + ".bin";
          // Vector2File(x, freq_data_file);
          } catch (std::runtime_error &e) {
            // Exception handling code
            std::cerr << "Exception caught: " << e.what() << std::endl;
          }
        }
      }
    }
    return 0;
}