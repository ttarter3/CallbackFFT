//
// Created by ttarter on 3/20/24.
//

#ifndef CUDATESTSTRING_GPUERRORCHK_H
#define CUDATESTSTRING_GPUERRORCHK_H

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#endif //CUDATESTSTRING_GPUERRORCHK_H
