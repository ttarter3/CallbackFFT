#ifndef FILLDATA_HPP
#define FILLDATA_HPP

// Standard Header
#include <random>
#include <type_traits>
#include <iomanip>

template<typename T>
void FillData(std::mt19937 & gen_ui, T & tmp, float min = -10, float max = 10) {
    std::uniform_real_distribution<T> distribution_d(min, max);
    tmp = distribution_d(gen_ui);
}


template <typename T, typename = std::enable_if_t<std::is_integral_v<T> && !std::is_floating_point_v<T>>>
void FillData(std::mt19937 & gen_ui, T & tmp, T min = 0, T max = 1<<11) {
  std::uniform_int_distribution<T> distribution_ui(min, max);
  tmp = distribution_ui(gen_ui);
}

template<typename T>
void FillData(std::mt19937 & gen_ui, std::vector<T> & tmp, T min=-10, T max=10) {
  #pragma omp parallel for
  for (std::size_t jj = 0; jj < tmp.size(); jj++) {
    FillData(gen_ui, tmp[jj], min, max);
  }; 
}

#endif // FILLDATA_HPP