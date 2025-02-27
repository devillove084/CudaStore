#pragma once
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#define CUDACHECK(ans) \
  { cudaAssert((ans), __FILE__, __LINE__); }
#define CUDASYNCCHECK() \
  { cudaAssert(cudaGetLastError(), __FILE__, __LINE__); }

inline const char* cudaErrorStringWithCode(cudaError_t error) {
  return cudaGetErrorName(error);
}

inline void cudaAssert(cudaError_t code, const char* file, int line) {
  if (code != cudaSuccess) {
    fprintf(stderr,
            "CUDA Error: %s\n\tCode: %d\n\tReason: %s\n\tLocation: %s:%d\n",
            cudaGetErrorString(code), static_cast<int>(code),
            cudaErrorStringWithCode(code), file, line);
    cudaDeviceReset();
    exit(EXIT_FAILURE);
  }
}
