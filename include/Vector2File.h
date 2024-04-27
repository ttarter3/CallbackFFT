//
// Created by ttarter on 3/21/24.
//

#ifndef CUDATESTSTRING_VECTOR2FILE_H
#define CUDATESTSTRING_VECTOR2FILE_H

#include <iostream>
#include <fstream>
#include <vector>

// Function to write vector data to a binary file
template<typename T>
void Vector2File(const std::vector<T>& vec, const std::string& filename) {
  std::ofstream file(filename, std::ios::out | std::ios::binary);
  if (file.is_open()) {
    // Writing the size of the vector first
    size_t size = vec.size();
    file.write(reinterpret_cast<const char*>(&size), sizeof(size));

    // Writing the vector elements
    file.write(reinterpret_cast<const char*>(&vec[0]), vec.size() * sizeof(T));

    file.close();
  } else {
    std::cerr << "Unable to open file: " << filename << std::endl;
  }
}

#endif //CUDATESTSTRING_VECTOR2FILE_H
