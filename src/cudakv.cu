#include <cuda_runtime.h>
#include <cstdio>

#include "cudakv/cuda_map.hpp"


namespace cudakv {
  __global__ void InsertKernel(GpuMap<int, int>* hashmap) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int key = tid % 512;
    hashmap->Insert(key, tid);
  }
  
  __global__ void QueryKernel(GpuMap<int, int>* hashmap, int* results) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int value;
    if (hashmap->Find(tid % 512, &value)) {
        results[tid] = value;
    } else {
        results[tid] = -1;
    }
  }
}

int run() {
  cudakv::GpuMap<int, int> hashmap(100);

  constexpr int NUM_THREADS = 256;
  constexpr int NUM_BLOCKS = 16;

  cudakv::InsertKernel<<<NUM_BLOCKS, NUM_THREADS>>> (&hashmap);
  cudaDeviceSynchronize();

  int* d_results;
  cudaMalloc(&d_results, NUM_THREADS * NUM_BLOCKS * sizeof(int));

  cudakv::QueryKernel<<<NUM_BLOCKS, NUM_THREADS>>>(&hashmap, d_results);

  int* h_results = new int[NUM_THREADS * NUM_BLOCKS];
  cudaMemcpy(h_results, d_results, NUM_THREADS * NUM_BLOCKS * sizeof(int),
             cudaMemcpyDeviceToHost);

  for (int i = 0; i < 10; ++i) {
    printf("Query key %d: result %d\n", i, h_results[i]);
  }

  delete[] h_results;
  cudaFree(d_results);
  return 0;
}