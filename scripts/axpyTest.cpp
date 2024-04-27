

// Parent Header
#include "axpy.hpp"

// Project Headers

// 3rd Party Headers

// Standard Headers
#include <vector>
#include <iostream>
#include <chrono>
#include <thread>

#define TYPE float



int main() {
	std::cout << "start" << std::endl;
  // std::this_thread::sleep_for(std::chrono::nanoseconds(10));
  // std::this_thread::sleep_until(std::chrono::system_clock::now() + std::chrono::seconds(1));
  
	TYPE a = 2.0f;
 
  int device, array_size;  
 
  device = 1;
  array_size = (float)4.0E9 / sizeof(TYPE); // x: 1000000000

  device = 0;
  //array_size = (float)4.2E9 / sizeof(TYPE); // x: 1050000000

	std::vector<TYPE> x(array_size, 0);
	std::vector<TYPE> y(x.size(), 0);
	
  std::cout << "x: " << x.size() << std::endl;
  std::cout << "y: " << y.size() << std::endl;
     
 
	for (int ii = 0; ii < x.size(); ii++) {
		x[ii] = ii;
	}


	Axpy<TYPE> axpy(x.size(), device);
	axpy.Load(x.data(), y.data());
	axpy.Execute(a);
	axpy.Purge(y.data());

	for (int ii = 1; ii < y.size(); ii*=2) {
		std::cout << "Value: (" << y[ii] << "),\tError: (" << y[ii] - ii * a << ")" << std::endl;
	};
	
	return 0;
}

/*

Number of GPUs: 2
Using device 1: NVIDIA T1000 8GB

Operating Time(millsec): 82.152100


Number of GPUs: 2
Using device 0: NVIDIA A2

Operating Time(millsec): 86.139999

Value: (2),     Error: (0)
Value: (4),     Error: (0)
Value: (8),     Error: (0)
Value: (16),    Error: (0)
Value: (32),    Error: (0)
Value: (64),    Error: (0)
Value: (128),   Error: (0)
Value: (256),   Error: (0)
Value: (512),   Error: (0)
Value: (1024),  Error: (0)
Value: (2048),  Error: (0)
Value: (4096),  Error: (0)
Value: (8192),  Error: (0)
Value: (16384), Error: (0)
Value: (32768), Error: (0)
Value: (65536), Error: (0)
Value: (131072),        Error: (0)
Value: (262144),        Error: (0)
Value: (524288),        Error: (0)
Value: (1.04858e+06),   Error: (0)
Value: (2.09715e+06),   Error: (0)
Value: (4.1943e+06),    Error: (0)
Value: (8.38861e+06),   Error: (0)
Value: (1.67772e+07),   Error: (0)
Value: (3.35544e+07),   Error: (0)
Value: (6.71089e+07),   Error: (0)
Value: (1.34218e+08),   Error: (0)
Value: (2.68435e+08),   Error: (0)
Value: (5.36871e+08),   Error: (0)
Value: (1.07374e+09),   Error: (0)
*/
