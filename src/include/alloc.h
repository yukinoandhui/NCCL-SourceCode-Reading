/*************************************************************************
 * Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_ALLOC_H_
#define NCCL_ALLOC_H_

#include "nccl.h"
#include "checks.h"
#include "bitops.h"
#include "utils.h"
#include "p2p.h"
#include <sys/mman.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>

#if CUDART_VERSION >= 11030
#include <cuda.h>
#include "cudawrap.h"
#endif

uint64_t clockNano(); // from utils.h with which we have a circular dependency

template<typename T>
constexpr size_t ncclSizeOfT() { return sizeof(T); }
template<>
constexpr size_t ncclSizeOfT<void>() { return 1; }

#if CUDART_VERSION >= 12020

static inline ncclResult_t ncclCuMemHostAlloc(void** ptr, CUmemGenericAllocationHandle *handlep, size_t size) {
  ncclResult_t result = ncclSuccess;
  size_t granularity = 0;
  CUdevice currentDev;
  CUmemAllocationProp prop = {};
  CUmemAccessDesc accessDesc = {};
  CUmemGenericAllocationHandle handle;
  int cudaDev;
  int cpuNumaNodeId = -1;
  CUmemAllocationHandleType type = ncclCuMemHandleType;

  CUDACHECK(cudaGetDevice(&cudaDev));
  CUCHECK(cuDeviceGet(&currentDev, cudaDev));
  CUCHECK(cuDeviceGetAttribute(&cpuNumaNodeId, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, currentDev));
  if (cpuNumaNodeId < 0) cpuNumaNodeId = 0;
  prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.requestedHandleTypes = type; // So it can be exported
  prop.location.id = cpuNumaNodeId;
  CUCHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  ALIGN_SIZE(size, granularity);
  /* Allocate the physical memory on the device */
  CUCHECK(cuMemCreate(&handle, size, &prop, 0));
  /* Reserve a virtual address range */
  CUCHECK(cuMemAddressReserve((CUdeviceptr*)ptr, size, granularity, 0, 0));
  /* Map the virtual address range to the physical allocation */
  CUCHECK(cuMemMap((CUdeviceptr)*ptr, size, 0, handle, 0));
  /* Now allow RW access to the newly mapped memory for local GPU */
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = cudaDev;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CUCHECK(cuMemSetAccess((CUdeviceptr)*ptr, size, &accessDesc, 1));

  /* Now allow RW access to the newly mapped memory from the CPU */
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
  accessDesc.location.id = cpuNumaNodeId;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CUCHECK(cuMemSetAccess((CUdeviceptr)*ptr, size, &accessDesc, 1));

  if (handlep) *handlep = handle;
  INFO(NCCL_ALLOC, "CUMEM Host Alloc Size %zi pointer %p handle %llx numa %d dev %d granularity %ld", size, *ptr, handle, cpuNumaNodeId, cudaDev, granularity);
  return result;
}

static inline ncclResult_t ncclCuMemHostFree(void* ptr) {
  if (ptr == NULL) return ncclSuccess;
  ncclResult_t result = ncclSuccess;
  CUmemGenericAllocationHandle handle;
  size_t size = 0;
  CUCHECK(cuMemRetainAllocationHandle(&handle, ptr));
  CUCHECK(cuMemRelease(handle));
  CUCHECK(cuMemGetAddressRange(NULL, &size, (CUdeviceptr)ptr));
  TRACE(NCCL_ALLOC, "CUMEM Host Free Size %zi pointer %p handle 0x%llx", size, ptr, handle);
  CUCHECK(cuMemUnmap((CUdeviceptr)ptr, size));
  CUCHECK(cuMemRelease(handle));
  CUCHECK(cuMemAddressFree((CUdeviceptr)ptr, size));
  return result;
}

#else /* CUDART_VERSION >= 12020 */

static inline ncclResult_t ncclCuMemHostAlloc(void** ptr, void* handlep, size_t size) {
  WARN("CUMEM Host is not supported prior to CUDA 12.2");
  return ncclInternalError;
}

static inline ncclResult_t ncclCuMemHostFree(void* ptr) {
  WARN("CUMEM Host is not supported prior to CUDA 12.2");
  return ncclInternalError;
}

#endif  /* CUDART_VERSION >= 12020 */

template <typename T>
ncclResult_t ncclCudaHostCallocDebug(T** ptr, size_t nelem, const char *filefunc, int line) {
  ncclResult_t result = ncclSuccess;
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  *ptr = nullptr;
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  if (nelem > 0) {
    CUDACHECKGOTO(cudaHostAlloc(ptr, nelem*ncclSizeOfT<T>(), cudaHostAllocMapped), result, finish);
    memset(*ptr, 0, nelem*ncclSizeOfT<T>());
  }
finish:
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  if (*ptr == nullptr && nelem > 0) WARN("Failed to CUDA host alloc %ld bytes", nelem*ncclSizeOfT<T>());
  INFO(NCCL_ALLOC, "%s:%d Cuda Host Alloc Size %ld pointer %p", filefunc, line, nelem*ncclSizeOfT<T>(), *ptr);
  return result;
}

static inline ncclResult_t ncclCudaHostFree(void* ptr) {
  CUDACHECK(cudaFreeHost(ptr));
  return ncclSuccess;
}

#define ncclCudaHostCalloc(...) ncclCudaHostCallocDebug(__VA_ARGS__, __FILE__, __LINE__)

template <typename T>
ncclResult_t ncclCallocDebug(T** ptr, size_t nelem, const char *filefunc, int line) {
  if (nelem > 0) {
    T* p = (T*)malloc(nelem*ncclSizeOfT<T>());
    if (p == NULL) {
      WARN("Failed to malloc %ld bytes", nelem*ncclSizeOfT<T>());
      return ncclSystemError;
    }
    //INFO(NCCL_ALLOC, "%s:%d malloc Size %ld pointer %p", filefunc, line, nelem*ncclSizeOfT<T>(), p);
    memset(p, 0, nelem*ncclSizeOfT<T>());
    *ptr = p;
  } else {
    *ptr = NULL;
  }
  return ncclSuccess;
}
#define ncclCalloc(...) ncclCallocDebug(__VA_ARGS__, __FILE__, __LINE__)

template <typename T>
ncclResult_t ncclRealloc(T** ptr, size_t oldNelem, size_t nelem) {
  T* oldp = *ptr;
  if (nelem < oldNelem || (oldp == NULL && oldNelem > 0)) return ncclInternalError;
  if (nelem == oldNelem) return ncclSuccess;

  T* p = (T*)malloc(nelem*ncclSizeOfT<T>());
  if (p == NULL) {
    WARN("Failed to malloc %ld bytes", nelem*ncclSizeOfT<T>());
    return ncclSystemError;
  }
  if (oldp && oldNelem) memcpy(p, oldp, oldNelem * ncclSizeOfT<T>());
  if (oldp) free(oldp);
  memset(p+oldNelem, 0, (nelem-oldNelem)*ncclSizeOfT<T>());
  *ptr = (T*)p;
  INFO(NCCL_ALLOC, "Mem Realloc old size %ld, new size %ld pointer %p", oldNelem*ncclSizeOfT<T>(), nelem*ncclSizeOfT<T>(), *ptr);
  return ncclSuccess;
}

#if CUDART_VERSION >= 11030

#include <cuda.h>
#include "cudawrap.h"

// ncclCuMemAllocAddr takes memory handle and size and returns the mapped address pointer
static inline ncclResult_t ncclCuMemAllocAddr(void **ptr, CUmemGenericAllocationHandle *handleIn, size_t size) {
  ncclResult_t result = ncclSuccess;
  size_t granularity = 0;
  CUmemAllocationProp prop = {};
  CUmemAccessDesc accessDesc = {};
  int cudaDev;
  CUDACHECK(cudaGetDevice(&cudaDev));
  CUCHECK(cuMemGetAllocationPropertiesFromHandle(&prop, *handleIn));
  CUCHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  ALIGN_SIZE(size, granularity);
  /* Reserve a virtual address range */
  CUCHECK(cuMemAddressReserve((CUdeviceptr *)ptr, size, granularity, 0, 0));
  /* Map the virtual address range to the physical allocation */
  CUCHECK(cuMemMap((CUdeviceptr)*ptr, size, 0, *handleIn, 0));
  /* Now allow RW access to the newly mapped memory */
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = cudaDev;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CUCHECK(cuMemSetAccess((CUdeviceptr)*ptr, size, &accessDesc, 1));
  TRACE(NCCL_ALLOC, "CuMem Map Size %zu pointer %p handle %llx", size, *ptr, *handleIn);
  return result;
}

static inline ncclResult_t ncclCuMemFreeAddr(void *ptr) {
  if (ptr == NULL) return ncclSuccess;
  ncclResult_t result = ncclSuccess;
  size_t size = 0;
  CUCHECK(cuMemGetAddressRange(NULL, &size, (CUdeviceptr)ptr));
  CUCHECK(cuMemUnmap((CUdeviceptr)ptr, size));
  CUCHECK(cuMemAddressFree((CUdeviceptr)ptr, size));
  return result;
}

static inline ncclResult_t ncclCuMemAlloc(void **ptr, CUmemGenericAllocationHandle *handlep, CUmemAllocationHandleType type, size_t size) {
  ncclResult_t result = ncclSuccess;
  size_t granularity = 0;
  CUdevice currentDev;
  CUmemAllocationProp prop = {};
  CUmemAccessDesc accessDesc = {};
  CUmemGenericAllocationHandle handle;
  int cudaDev;
  int flag = 0;
  CUDACHECK(cudaGetDevice(&cudaDev));
  CUCHECK(cuDeviceGet(&currentDev, cudaDev));
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.requestedHandleTypes = type;
  prop.location.id = currentDev;
  // Query device to see if RDMA support is available
  CUCHECK(cuDeviceGetAttribute(&flag, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED, currentDev));
  if (flag) prop.allocFlags.gpuDirectRDMACapable = 1;
  CUCHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  ALIGN_SIZE(size, granularity);
  /* Allocate the physical memory on the device */
  CUCHECK(cuMemCreate(&handle, size, &prop, 0));
  /* Reserve a virtual address range */
  CUCHECK(cuMemAddressReserve((CUdeviceptr *)ptr, size, granularity, 0, 0));
  /* Map the virtual address range to the physical allocation */
  CUCHECK(cuMemMap((CUdeviceptr)*ptr, size, 0, handle, 0));
  /* Now allow RW access to the newly mapped memory */
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = currentDev;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CUCHECK(cuMemSetAccess((CUdeviceptr)*ptr, size, &accessDesc, 1));
  if (handlep) *handlep = handle;
  TRACE(NCCL_ALLOC, "CuMem Alloc Size %zu pointer %p handle %llx", size, *ptr, handle);
  return result;
}

static inline ncclResult_t ncclCuMemFree(void *ptr) {
  if (ptr == NULL) return ncclSuccess;
  ncclResult_t result = ncclSuccess;
  CUmemGenericAllocationHandle handle;
  size_t size = 0;
  CUCHECK(cuMemRetainAllocationHandle(&handle, ptr));
  CUCHECK(cuMemRelease(handle));
  CUCHECK(cuMemGetAddressRange(NULL, &size, (CUdeviceptr)ptr));
  TRACE(NCCL_ALLOC, "CuMem Free Size %zu pointer %p handle 0x%llx", size, ptr, handle);
  CUCHECK(cuMemUnmap((CUdeviceptr)ptr, size));
  CUCHECK(cuMemRelease(handle));
  CUCHECK(cuMemAddressFree((CUdeviceptr)ptr, size));
  return result;
}

#else

extern int ncclCuMemEnable();

static inline ncclResult_t ncclCuMemAlloc(void **ptr, void *handlep, int type, size_t size) {
  WARN("CUMEM not supported prior to CUDA 11.3");
  return ncclInternalError;
}
static inline ncclResult_t ncclCuMemFree(void *ptr) {
  WARN("CUMEM not supported prior to CUDA 11.3");
  return ncclInternalError;
}

static inline ncclResult_t ncclCuMemAllocAddr(void **ptr, CUmemGenericAllocationHandle *handleIn, size_t size) {
  WARN("CUMEM not supported prior to CUDA 11.3");
  return ncclInternalError;
}

static inline ncclResult_t ncclCuMemFreeAddr(void *ptr) {
  WARN("CUMEM not supported prior to CUDA 11.3");
  return ncclInternalError;
}
#endif

template <typename T>
ncclResult_t ncclCudaMallocDebug(T** ptr, size_t nelem, const char *filefunc, int line) {
  ncclResult_t result = ncclSuccess;
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;//设置流捕获模式为 relaxed
  *ptr = nullptr;
  /*
  在多线程编程中，当多个线程需要​​协同捕获同一个 CUDA 流​​时，cudaThreadExchangeStreamCaptureMode 定义了线程之间如何传递和同步流捕获状态。
  */
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));//// 设置新模式，mode变量被更新为旧模式  
  //// 执行需要临时模式的代码  
  if (nelem > 0) {
    if (ncclCuMemEnable()) {
      NCCLCHECKGOTO(ncclCuMemAlloc((void **)ptr, NULL, ncclCuMemHandleType, nelem*ncclSizeOfT<T>()), result, finish);
    } else {
    }
    CUDACHECKGOTO(cudaMalloc(ptr, nelem*ncclSizeOfT<T>()), result, finish);
  }
finish:
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));//// 恢复原始模式  
  if (*ptr == nullptr && nelem > 0) WARN("Failed to CUDA malloc %ld bytes", nelem*ncclSizeOfT<T>());
  INFO(NCCL_ALLOC, "%s:%d Cuda Alloc Size %ld pointer %p", filefunc, line, nelem*ncclSizeOfT<T>(), *ptr);
  return result;
}
#define ncclCudaMalloc(...) ncclCudaMallocDebug(__VA_ARGS__, __FILE__, __LINE__)

template <typename T>
ncclResult_t ncclCudaCallocDebug(T** ptr, size_t nelem, const char *filefunc, int line) {
  ncclResult_t result = ncclSuccess;
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  *ptr = nullptr;
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  if (nelem > 0) {
    // Need a side stream so as not to interfere with graph capture.
    cudaStream_t stream;
    CUDACHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    if (ncclCuMemEnable()) {
      NCCLCHECKGOTO(ncclCuMemAlloc((void **)ptr, NULL, ncclCuMemHandleType, nelem*ncclSizeOfT<T>()), result, finish);
    } else {
      CUDACHECKGOTO(cudaMalloc(ptr, nelem*ncclSizeOfT<T>()), result, finish);
    }
    CUDACHECKGOTO(cudaMemsetAsync(*ptr, 0, nelem*ncclSizeOfT<T>(), stream), result, finish);
    CUDACHECKGOTO(cudaStreamSynchronize(stream), result, finish);
    CUDACHECKGOTO(cudaStreamDestroy(stream), result, finish);
  }
finish:
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  if (*ptr == nullptr && nelem > 0) WARN("Failed to CUDA calloc %ld bytes", nelem*ncclSizeOfT<T>());
  INFO(NCCL_ALLOC, "%s:%d Cuda Alloc Size %ld pointer %p", filefunc, line, nelem*ncclSizeOfT<T>(), *ptr);
  return result;
}
#define ncclCudaCalloc(...) ncclCudaCallocDebug(__VA_ARGS__, __FILE__, __LINE__)

template <typename T>
ncclResult_t ncclCudaCallocAsyncDebug(T** ptr, size_t nelem, cudaStream_t stream, const char *filefunc, int line) {
  ncclResult_t result = ncclSuccess;
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  *ptr = nullptr;
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  if (nelem > 0) {
    if (ncclCuMemEnable()) {
      NCCLCHECKGOTO(ncclCuMemAlloc((void **)ptr, NULL, ncclCuMemHandleType, nelem*ncclSizeOfT<T>()), result, finish);
    } else {
      CUDACHECKGOTO(cudaMalloc(ptr, nelem*ncclSizeOfT<T>()), result, finish);//不会捕获，但是Relaxed模式会允许执行
    }
    CUDACHECKGOTO(cudaMemsetAsync(*ptr, 0, nelem*ncclSizeOfT<T>(), stream), result, finish);
  }
finish:
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  if (*ptr == nullptr && nelem > 0) WARN("Failed to CUDA calloc async %ld bytes", nelem*ncclSizeOfT<T>());
  INFO(NCCL_ALLOC, "%s:%d Cuda Alloc Size %ld pointer %p", filefunc, line, nelem*ncclSizeOfT<T>(), *ptr);
  return result;
}
#define ncclCudaCallocAsync(...) ncclCudaCallocAsyncDebug(__VA_ARGS__, __FILE__, __LINE__)

template <typename T>
ncclResult_t ncclCudaMemcpy(T* dst, T* src, size_t nelem) {
  ncclResult_t result = ncclSuccess;
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  // Need a side stream so as not to interfere with graph capture.
  cudaStream_t stream;
  CUDACHECKGOTO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), result, finish);
  NCCLCHECKGOTO(ncclCudaMemcpyAsync(dst, src, nelem, stream), result, finish);
  CUDACHECKGOTO(cudaStreamSynchronize(stream), result, finish);
  CUDACHECKGOTO(cudaStreamDestroy(stream), result, finish);
finish:
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  return result;
}

template <typename T>
ncclResult_t ncclCudaMemcpyAsync(T* dst, T* src, size_t nelem, cudaStream_t stream) {
  ncclResult_t result = ncclSuccess;
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  CUDACHECKGOTO(cudaMemcpyAsync(dst, src, nelem*ncclSizeOfT<T>(), cudaMemcpyDefault, stream), result, finish);
finish:
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  return result;
}

template <typename T>
ncclResult_t ncclCudaFree(T* ptr) {
  ncclResult_t result = ncclSuccess;
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  TRACE(NCCL_ALLOC, "Cuda Free pointer %p", ptr);
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  if (ncclCuMemEnable()) {
    NCCLCHECKGOTO(ncclCuMemFree((void *)ptr), result, finish);
  } else {
    CUDACHECKGOTO(cudaFree(ptr), result, finish);
  }
finish:
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  return result;
}

// Allocate memory to be potentially ibv_reg_mr'd. This needs to be
// allocated on separate pages as those pages will be marked DONTFORK
// and if they are shared, that could cause a crash in a child process
inline ncclResult_t ncclIbMallocDebug(void** ptr, size_t size, const char *filefunc, int line) {
  if (size > 0) {
    long page_size = sysconf(_SC_PAGESIZE);
    if (page_size < 0) return ncclSystemError;
    void* p;
    int size_aligned = ROUNDUP(size, page_size);
    int ret = posix_memalign(&p, page_size, size_aligned);
    if (ret != 0) return ncclSystemError;
    memset(p, 0, size);
    *ptr = p;
  } else {
    *ptr = NULL;
  }
  INFO(NCCL_ALLOC, "%s:%d Ib Alloc Size %ld pointer %p", filefunc, line, size, *ptr);
  return ncclSuccess;
}
#define ncclIbMalloc(...) ncclIbMallocDebug(__VA_ARGS__, __FILE__, __LINE__)

#endif
