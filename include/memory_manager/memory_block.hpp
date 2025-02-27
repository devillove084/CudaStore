#pragma once

#include <cstddef>

namespace cudakv {
  struct MemoryBlock {
    void* ptr = nullptr;
    std::size_t size;
    int device_id;
    bool is_gpu_memory;
  };
}