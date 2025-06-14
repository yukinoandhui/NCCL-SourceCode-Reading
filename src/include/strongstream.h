/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_STRONGSTREAM_H_
#define NCCL_STRONGSTREAM_H_

#include "nccl.h"
#include "checks.h"

#include <stdint.h>

/* ncclCudaGraph: Wraps a cudaGraph_t so that we can support pre-graph CUDA runtimes
 * easily.
 */
struct ncclCudaGraph {
#if CUDART_VERSION >= 11030
  cudaGraph_t graph;
  unsigned long long graphId;
#endif
};

inline struct ncclCudaGraph ncclCudaGraphNone() {
  struct ncclCudaGraph tmp;
  #if CUDART_VERSION >= 11030
    tmp.graph = nullptr;
    tmp.graphId = ULLONG_MAX;
  #endif
  return tmp;
}

inline bool ncclCudaGraphValid(struct ncclCudaGraph graph) {
  #if CUDART_VERSION >= 11030
    return graph.graph != nullptr;
  #else
    return false;
  #endif
}

inline bool ncclCudaGraphSame(struct ncclCudaGraph a, struct ncclCudaGraph b) {
  #if CUDART_VERSION >= 11030
    return a.graphId == b.graphId;
  #else
    return true;
  #endif
}

ncclResult_t ncclCudaGetCapturingGraph(struct ncclCudaGraph* graph, cudaStream_t stream);
ncclResult_t ncclCudaGraphAddDestructor(struct ncclCudaGraph graph, cudaHostFn_t fn, void* arg);

/* ncclStrongStream: An abstraction over CUDA streams that do not lose their
 * identity while being captured. Regular streams have the deficiency that the
 * captured form of a stream in one graph launch has no relation to the
 * uncaptured stream or to the captured form in other graph launches. This makes
 * streams unfit for the use of serializing access to a persistent resource.
 * Strong streams have been introduced to address this need.
 *
 * - All updates to a strong stream must be enclosed by a Acquire/Release pair.
 *
 * - The Acquire, Release, and all updates take a ncclCudaGraph parameter
 *   indicating the currently capturing graph (or none). This parameter must be
 *   the same for the entire sequence of {Acquire; ...; Release}.
 *
 * - An {Acquire; ...; Release} sequence must not be concurrent with any
 *   other operations against the strong stream including graph launches which
 *   reference this stream.
 */
struct ncclStrongStream;

ncclResult_t ncclStrongStreamConstruct(struct ncclStrongStream* ss);
ncclResult_t ncclStrongStreamDestruct(struct ncclStrongStream* ss);

// Acquire-fence the strong stream.
ncclResult_t ncclStrongStreamAcquire(
  struct ncclCudaGraph graph, struct ncclStrongStream* ss
);

// Acquire-fence the strong stream assuming no graph is capturing. This permits
// the caller to enqueue directly to the `ss->cudaStream` member using native CUDA
// calls. Strong stream still must be released via:
//   ncclStrongStreamRelease(ncclCudaGraphNone(), ss);
ncclResult_t ncclStrongStreamAcquireUncaptured(struct ncclStrongStream* ss);

// Release-fence of the strong stream.
ncclResult_t ncclStrongStreamRelease(struct ncclCudaGraph graph, struct ncclStrongStream* ss);

// Add a host launch to the stream.
ncclResult_t ncclStrongStreamLaunchHost(
  struct ncclCudaGraph graph, struct ncclStrongStream* ss,
  cudaHostFn_t fn, void* arg
);
// Add a kernel launch to the stream.
ncclResult_t ncclStrongStreamLaunchKernel(
  struct ncclCudaGraph graph, struct ncclStrongStream* ss,
  void* fn, dim3 grid, dim3 block, void** args, size_t sharedMemBytes
);

// Cause `a` to wait for the current state `b`. Both `a` and `b` must be acquired.
// `b_subsumes_a` indicates that all work in `a` is already present in `b`, thus
// we want to fast-forward `a` to be a clone of `b`. Knowing this permits the
// implementation to induce few graph dependencies.
ncclResult_t ncclStrongStreamWaitStream(
  struct ncclCudaGraph graph, struct ncclStrongStream* a, struct ncclStrongStream* b, bool b_subsumes_a=false
);
// `b` must be capturing within `graph`.
ncclResult_t ncclStrongStreamWaitStream(
  struct ncclCudaGraph graph, struct ncclStrongStream* a, cudaStream_t b, bool b_subsumes_a=false
);
// `a` must be capturing within `graph`.
ncclResult_t ncclStrongStreamWaitStream(
  struct ncclCudaGraph graph, cudaStream_t a, struct ncclStrongStream* b, bool b_subsumes_a=false
);

// Synchrnoization does not need the strong stream to be acquired.
ncclResult_t ncclStrongStreamSynchronize(struct ncclStrongStream* ss);

////////////////////////////////////////////////////////////////////////////////

struct ncclStrongStreamGraph; // internal to ncclStrongStream

/*
假设一个 CUDA stream 先被用于 Graph 捕获（即把一系列操作录制成 CUDA Graph），然后又被用于普通的 kernel 或 NCCL 通信操作。如果不做特殊处理，普通操作可能会在 Graph 还没执行完时就插入到 stream 里，导致执行顺序混乱（比如数据还没准备好就被用来通信）。

为了解决这个竞态，NCCL 采用了事件（serialEvent）机制
- 在 Graph 捕获结束时，记录一个 event（serialEvent）。
- 后续每次在该 stream 上添加普通操作前，先让 stream 等待这个 event（ cudaStreamWaitEvent ），确保 Graph 捕获的内容已经执行完毕。
  这样就保证了 Graph 和普通操作之间的严格先后顺序。
- Acquire ：表示 NCCL/用户即将要在这个 stream 上添加新的操作（比如 kernel、通信等），需要先做好同步准备。此时会检查是否需要等待 serialEvent。
- Release ：表示 NCCL/用户已经在 stream 上添加完操作，需要在 stream 上记录一个新的 event（serialEvent），以便下次 Acquire 时可以等待这个新的同步点。

如果一个 stream 既被 Graph 捕获过，又要继续用于普通的 kernel/通信操作，就需要保证两者之间的执行顺序不会乱。
如果 stream 曾经被 Graph 捕获过（ everCaptured ），
那么 NCCL 需要让后续在该 stream 上的操作等待 Graph 捕获时的结束点（ serialEvent ），以保证不会出现“Graph 还没执行完，普通操作就插入进来”的竞态。
*/
struct ncclStrongStream {
  // Used when not graph capturing.
  cudaStream_t cudaStream;
#if CUDART_VERSION >= 11030
  // The event used to establish order between graphs and streams. During acquire
  // this event is waited on, during release it is recorded to. 
  //用于在 CUDA Graph capture 和普通 stream 之间建立顺序依赖。acquire 时等待该 event，release 时记录 event，确保操作顺序。
  cudaEvent_t serialEvent;
  // This stream ever appeared in a graph capture. 标记该 stream 是否曾经被 CUDA Graph 捕获过
  bool everCaptured; 
  // Tracks whether serialEvent needs to be recorded to upon Release().
  // 标记在 release 时是否需要向 stream 记录 serialEvent，通常在 acquire 后假定会有新工作加入。其实就是一个标记，用于判断是否需要记录 serialEvent。
  bool serialEventNeedsRecord; 
  struct ncclStrongStreamGraph* graphHead;
#else
  cudaEvent_t scratchEvent;//如果 CUDA 版本低于 11.3，仅有 cudaEvent_t scratchEvent; ，用于基本的事件同步
#endif
};

#endif
