#include "memory_manager/memory_scheduler.hpp"

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <iostream>
#include <mutex>

#include "common.hpp"

namespace cudakv {

void MemoryScheduler::Initialize(std::size_t init_gpu_pool_size) {
  std::lock_guard<std::mutex> guard(mutex_);

  int device_count = 0;
  CUDACHECK(cudaGetDeviceCount(&device_count));
  gpu_pools_.resize(device_count);

  for (auto dev = 0; dev < device_count; ++dev) {
    CUDACHECK(cudaSetDevice(dev));
    gpu_pools_[dev].total_size = init_gpu_pool_size;
    CUDACHECK(
        cudaMalloc(&gpu_pools_[dev].blocks[nullptr].ptr, init_gpu_pool_size));
  }
}

void* MemoryScheduler::Allocate(size_t bytes, int prefer_device) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (prefer_device < gpu_pools_.size()) {
    auto& pool = gpu_pools_[prefer_device];
    if (pool.used_size + bytes <= pool.total_size) {
      CUDACHECK(cudaSetDevice(prefer_device));
      void* ptr = nullptr;
      CUDACHECK(cudaMalloc(&ptr, bytes));
      if (ptr) {
        pool.blocks[ptr] = {ptr, bytes, prefer_device, true};
        pool.used_size += bytes;
        return ptr;
      }
    }
  }

  for (size_t dev = 0; dev < gpu_pools_.size(); ++dev) {
    if (dev == static_cast<size_t>(prefer_device)) continue;

    auto& pool = gpu_pools_[dev];
    if (pool.used_size + bytes <= pool.total_size) {
      CUDACHECK(cudaSetDevice(dev));
      void* ptr = nullptr;
      CUDACHECK(cudaMalloc(&ptr, bytes));
      if (ptr) {
        pool.blocks[ptr] = {ptr, bytes, static_cast<int>(dev), true};
        pool.used_size += bytes;
        return ptr;
      }
    }
  }

  void* host_ptr = malloc(bytes);
  if (host_ptr) {
    host_pool_[host_ptr] = {host_ptr, bytes, -1, false};
    host_pool_size_ += bytes;
    std::cout << "Allocated " << bytes << " bytes in host memory\n";
    return host_ptr;
  }

  throw std::bad_alloc();
}

void MemoryScheduler::Deallocate(void* ptr) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (host_pool_.count(ptr)) {
    free(ptr);
    host_pool_size_ -= host_pool_[ptr].size;
    host_pool_.erase(ptr);
    return;
  }

  for (auto& pool : gpu_pools_) {
    if (pool.blocks.count(ptr)) {
      CUDACHECK(cudaSetDevice(pool.blocks[ptr].device_id));
      CUDACHECK(cudaFree(ptr));
      pool.used_size -= pool.blocks[ptr].size;
      pool.blocks.erase(ptr);
      return;
    }
  }

  throw std::runtime_error("Trying to free unknown pointer");
}

MemoryScheduler::~MemoryScheduler() {
  for (auto& pool : gpu_pools_) {
    CUDACHECK(cudaSetDevice(pool.blocks.begin()->second.device_id));
    for (auto& [ptr, block] : pool.blocks) {
      CUDACHECK(cudaFree(ptr));
    }
  }
  for (auto& [ptr, block] : host_pool_) {
    free(ptr);
  }
}

}  // namespace cudakv