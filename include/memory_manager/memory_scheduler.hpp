#pragma once

#include <cstddef>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "memory_manager/device_memory_pool.hpp"
#include "memory_manager/memory_block.hpp"

namespace cudakv {
class MemoryScheduler {
 public:
  static MemoryScheduler& Instance() {
    static MemoryScheduler instance;
    return instance;
  }

  void Initialize(std::size_t init_gpu_pool_size);

  void* Allocate(std::size_t bytes, int prefer_device);

  void Deallocate(void* ptr);

 private:
  MemoryScheduler() = default;
  ~MemoryScheduler();

 private:
  std::vector<DeviceMemoryPool> gpu_pools_;
  std::unordered_map<void*, MemoryBlock> host_pool_;
  std::mutex mutex_;
  std::size_t host_pool_size_;
};
}  // namespace cudakv