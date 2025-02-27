#pragma once

#include <cstddef>
#include <unordered_map>

#include "memory_manager/memory_block.hpp"

namespace cudakv {

struct DeviceMemoryPool {
  std::unordered_map<void *, MemoryBlock> blocks;
  std::size_t total_size;
  std::size_t used_size;
};
}  // namespace cudakv