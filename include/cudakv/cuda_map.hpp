#pragma once

#include <cstddef>
#include <cuda_runtime.h>

// #include "common.hpp"

namespace cudakv {

enum class EntryStatus : int { EMPTY = 0, OCCUPIED = 1, DELETED = 2 };

template <typename Key, typename Value>
struct __align__(4) Entry {
  Key key;
  Value value;
  EntryStatus status;
};

template <typename Key, typename Value>
class GpuMap {
 public:
  // DISALLOW_COPY_AND_ASSIGN(GpuMap);

  __host__ GpuMap(std::size_t capacity);

  __host__ ~GpuMap();

  __device__ Entry<Key, Value>* DevicePtr();

  __device__ int Hash(const Key& key);

  __device__ bool Insert(const Key& key, const Value& value);

  __device__ bool Find(const Key& key, Value* out_value = nullptr);

  __host__ void DebugPrint();

  Entry<Key, Value>* entries_;
  std::size_t capacity_;
};

}  // namespace cudakv