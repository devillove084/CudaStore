#include "cudakv/cuda_map.hpp"

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_types.h>

#include <cstddef>
#include <cuda/atomic>

#include "common.hpp"

namespace cudakv {

template <typename Key, typename Value>
__host__ GpuMap<Key, Value>::GpuMap(std::size_t capacity_)
    : capacity_(capacity_) {
  CUDACHECK(cudaMalloc(&entries_, capacity_ * sizeof(Entry<Key, Value>)));
  CUDACHECK(cudaMemset(entries_, 0, capacity_ * sizeof(Entry<Key, Value>)));
}

template <typename Key, typename Value>
__host__ GpuMap<Key, Value>::~GpuMap() {
  CUDACHECK(cudaFree(entries_));
}

template <typename Key, typename Value>
__device__ int GpuMap<Key, Value>::Hash(const Key& key) {
  // ! just simple
  return reinterpret_cast<const int&>(key) % capacity_;
}

template <typename Key, typename Value>
__device__ bool GpuMap<Key, Value>::Insert(const Key& key, const Value& value) {
  Entry<Key, Value>* table = DevicePtr();
  int index = Hash(key);

  for (int i = 0; i < capacity_; ++i) {
    int current_idx = (index + i) % capacity_;

    int expected = static_cast<int>(EntryStatus::EMPTY);
    int* address = reinterpret_cast<int*>(&(table[current_idx].status));

    cuda::atomic<int, cuda::thread_scope_device>* status_ptr;

    if (status_ptr->compare_exchange_strong(
            expected, static_cast<int>(EntryStatus::OCCUPIED))) {
      table[current_idx].key = key;
      table[current_idx].value = value;
      return true;
    }

    if (table[current_idx].status == EntryStatus::OCCUPIED &&
        table[current_idx].key == key) {
      // TODO: Already exist
      return false;
    }
  }
  return false;
}

template <typename Key, typename Value>
__device__ bool GpuMap<Key, Value>::Find(const Key& key, Value* out_value) {
  Entry<Key, Value>* table = DevicePtr();
  int index = Hash(key);

  for (int i = 0; i < capacity_; ++i) {
    int current_idx = (index + i) % capacity_;

    if (table[current_idx].status == EntryStatus::EMPTY) {
      return false;
    }

    if (table[current_idx].status == EntryStatus::OCCUPIED &&
        table[current_idx].key == key) {
      if (out_value) *out_value = table[current_idx].value;
      return true;
    }
  }
  return false;
}

template <typename Key, typename Value>
__host__ void GpuMap<Key, Value>::DebugPrint() {
  Entry<Key, Value>* host_copy = new Entry<Key, Value>[capacity_];
  cudaMemcpy(host_copy, entries_, capacity_ * sizeof(Entry<Key, Value>),
             cudaMemcpyDeviceToHost);

  printf("Hash Table Contents:\n");
  for (int i = 0; i < capacity_; ++i) {
    if (host_copy[i].status == EntryStatus::OCCUPIED) {
      printf("[%d] Key: %d, Value: %d\n", i, host_copy[i].key,
             host_copy[i].value);
    }
  }
  delete[] host_copy;
}

template <typename Key, typename Value>
__host__ Entry<Key, Value>* GpuMap<Key, Value>::DevicePtr() {
  return entries_;
}

}  // namespace cudakv