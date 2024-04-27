#ifndef INCLUDE_LOADTWIDDLEFACTOR_HPP
#define INCLUDE_LOADTWIDDLEFACTOR_HPP

// Function to compute twiddle factors
template<typename T>
T Twiddle(int n, int N) {
    float angle = -2.0f * PI * (float)n / (float)N;
    return make_float2(cos(angle), sin(angle));
}


template<typename T>
void LoadTwiddleFactor(const int N) {
    T twiddle_factors[N];
    for (int n = 0; n < N; ++n) {
        twiddle_factors[n] = Twiddle<T>(n, N);
    }

    // Copy twiddle factors to constant memory
    cudaMemcpyToSymbol("twiddle_factors", twiddle_factors, N * sizeof(T));
}

#endif // INCLUDE_LOADTWIDDLEFACTOR_HPP