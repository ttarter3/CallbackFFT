//
// Created by ttarter on 3/20/24.
//

#ifndef CUDATESTSTRING_FINDINDEXOFMAX_H
#define CUDATESTSTRING_FINDINDEXOFMAX_H

// Standard Headers
#include <iostream>
#include <vector>
#include <cmath>

template <typename T>
size_t findIndexOfMax(const std::vector<std::complex<T>> & vec) {
    if (vec.empty()) {
        std::cerr << "Vector is empty!" << std::endl;
        return -1; // Return an invalid index
    }

    float maxVal = -INFINITY;
    size_t maxIndex = 0;

    for (size_t i = 0; i < vec.size(); ++i) {
        // Compute the magnitude of the float2 vector
        float magnitude = std::sqrt(vec[i].real() * vec[i].real() + vec[i].imag() * vec[i].imag());
        if (magnitude > maxVal) {
            maxVal = magnitude;
            maxIndex = i;
        }
    }

    return maxIndex;
}

#endif //CUDATESTSTRING_FINDINDEXOFMAX_H
