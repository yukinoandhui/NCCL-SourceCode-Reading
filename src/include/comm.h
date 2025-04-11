/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_COMM_H_
#define NCCL_COMM_H_

//#include "transport.h"
#include "p2p.h"
#include "collectives.h"
#include "nccl_tuner.h"
#include "proxy.h"
#include "strongstream.h"
#include "nccl_net.h"
#include "register.h"
#include "graph.h"
#include "profiler.h"

#if CUDART_VERSION < 9000
struct cudaLaunchParams {
  void *func;
  dim3 gridDim;
  dim3 blockDim;
  void **args;
  size_t sharedMem;
  cudaStream_t stream;
};
#endif

#define CACHE_LINE_SIZE 128 //cpu典型的Cache Line的大小一般是32，64 或者 128 字节。 gpu的cache line大小一般是128字节
#define MEM_ALIGN 4096 //页面大小
#define CUDA_IPC_MIN 2097152UL

// Channels / LL tuning
#define NCCL_LL_THREAD_THRESHOLD 8 //小消息使用更少的线程（8个线程）
#define NCCL_LL128_THREAD_THRESHOLD 8
#define NCCL_SIMPLE_THREAD_THRESHOLD 64



/*
// 使用环形缓冲区，支持流水线通信；缓冲区大小固定，避免了频繁的内存分配和释放。
// head和tail指针用于跟踪缓冲区的使用情况。类似于读写指针。
  NCCL_STEPS被理解为通信步骤。例如这里为8，那么我调用了八次all reduce，这八个就可以并行。
  如果 NCCL_STEPS 过小（例如 1-2），可能无法充分利用 DMA 并行能力，导致带宽利用率低。
  如果 NCCL_STEPS 过大，可能会导致 GPU 端的 buffer 资源消耗过多，影响其他计算任务。
*/
struct ncclSendMem {
  union {
    struct {
      uint64_t head; // 头部位置，接收方通过检查这个值来确定有多少数据已经被发送
      char pad1[CACHE_LINE_SIZE-sizeof(uint64_t)];//对齐，使得ptrExchange和缓存行对齐
      void* ptrExchange; //用于交换指针的地址
      uint64_t redOpArgExchange[2]; //用于交换归约操作的参数。在执行归约操作（如sum、max等）时，可能需要传递额外的参。
      char pad2[CACHE_LINE_SIZE-sizeof(void*)-2*sizeof(uint64_t)];
      int offsFifo[NCCL_STEPS];//偏移量先进先出队列，存储每个通信步骤的缓冲区偏移量，实现环形缓冲区机制，支持流水线通信
    };
    char pad3[MEM_ALIGN];// 
  };
};

struct ncclRecvMem {
  union {
    struct {
      uint64_t tail; //指示接收缓冲区的尾部位置。
      char pad1[CACHE_LINE_SIZE-sizeof(uint64_t)];
      struct ncclConnFifo connFifo[NCCL_STEPS];//一个连接队列（ncclConnFifo 结构数组），存储每个通信步骤的连接状态（如数据包元信息、错误码）。
      int flush; // 用于强制刷新缓存（如 GPU 显存到主机内存），确保数据一致性。在基于 ​GDRCopy（GPU Direct RDMA）的通信中，显存数据可能绕过 CPU 直接传输，需通过 flush 同步。
    };
    char pad4[MEM_ALIGN];
  };
};

enum helperThreadState {ThreadStart, ThreadStop};

#define NCCL_IPC_POOL_SIZE (2*NCCL_MAX_LOCAL_RANKS*NCCL_MAX_OPS)
// IPC池
struct ncclGraphHelperResources {
  ncclComm* comm;
  pthread_mutex_t threadLock;
  pthread_cond_t  threadCond;
  enum helperThreadState threadState;
  void* ipcBases[NCCL_IPC_POOL_SIZE];
  int ipcTail;
  int ipcHead; // 与 ipcTail 一起，用于实现环形缓冲区的机制。
};

struct ncclUserRedOp {
  int freeNext; // -1=allocated, otherwise index of next free entry in array
  ncclDataType_t datatype;
  ncclDevRedOpFull opFull;
};
/*
例如在8卡分布式训练（2个节点，每个节点4卡）中：

节点0的映射关系：
localRank: 0,1,2,3 -> globalRank: 0,1,2,3

节点1的映射关系：
localRank: 0,1,2,3 -> globalRank: 4,5,6,7
 
*/
struct ncclNodeRanks {
  int localRanks; //当前计算节点上的GPU数量。
  int* localRankToRank; // 用于将本地GPU的rank映射到全局rank。
};
//用于管理 NCCL 中的集群（clique）信息，主要用于 MNNVL（Multi-Node NVLink）场景。
struct cliqueInfo {
  int id;//用于区分不同的 NVLink 互联集群。
  int size; //表示有多少个 GPU 通过 NVLink 直接互联
  int *ranks;// 存储集群中所有 GPU 的全局 rank 数组。用于记录哪些 GPU 属于这个集群
};
//NCCL 中用于管理资源清理的机制，它实现了一个析构器链表。
struct ncclDestructor {
  struct ncclDestructor* next;//下一个要析构的资源。
  void* obj;// 指向需要被清理的对象。
  ncclResult_t(*fn)(struct ncclDestructor* me);// 指向实际执行清理工作的函数
};
//NCCL 中用于实现异步回调机制的基础结构。
struct ncclCommCallback {
  struct ncclCommCallback* next;
  ncclResult_t(*fn)(struct ncclComm* comm, struct ncclCommCallback* cb);
};
// NCCL 中用于处理 CUDA 事件相关回调的机制。
struct ncclCommEventCallback {
  struct ncclCommEventCallback* next;
  cudaEvent_t event;
  ncclResult_t(*fn)(struct ncclComm* comm, struct ncclCommEventCallback* cb);
};

//管理 NCCL 中的共享资源。
struct ncclSharedResources {
  int refCount; //引用计数，跟踪有多少通信器在使用这些共享资源
  struct ncclComm* owner; /* comm which creates this shared res. */ //创建这些共享资源的通信器。
  // NCCL 中将 MAXCHANNELS 设置为 64 是一个上限值。
  struct ncclChannelPeer* peers[MAXCHANNELS]; // 每个通道的对等点信息  CPU 端对等点数组
  struct ncclDevChannelPeer* devPeers[MAXCHANNELS]; //设备端的对等点信息 GPU 端对等点数组
  /* P2P operation counter, one per channel */
  uint64_t p2pOpCount[MAXCHANNELS]; //每个通道的点对点操作计数器
  /* Collective operation counter */
  uint64_t collOpCount; // 用于记录所有 collective 操作的计数器。
  // 当一个 NCCL 通信器被分割成多个子通信器时，原始的通信器被称为 "top parent"、这种设计允许子通信器共享原始通信器的某些资源。
  int tpNRanks; //顶级父级通信器中的总rank数
  int tpNLocalRanks; //本地 rank 数量
  int tpNChannels;//通道数量
  int tpP2pNChannels;//点对点通信通道数量
  int tpP2pChunkSize; //点对点通信的数据块大小
  uint64_t magic;

  // top parent rank to localRank translation table
  int* tpRankToLocalRank;//rank 到本地 rank 的映射表
  // Internal streams
  struct ncclStrongStream deviceStream, hostStream; //GPU 设备流、主机端流

   /*
    - 管理跨设备通信的代理线程
    - 处理 GPU 与网络接口之间的数据传输
    - 协调不同通信通道之间的资源分配
   */
  /* proxy related shared res */
  struct ncclProxyState* proxyState; //代理相关的共享资源.
};

struct ncclChannel {
  // peer是指的是 GPU 设备之间的点对点通信通道。
  struct ncclChannelPeer** peers;
  struct ncclDevChannelPeer** devPeers;
  /* devPeer pointer array used for host side access */
  struct ncclDevChannelPeer** devPeersHostPtr;
  // 通信拓扑结构
  struct ncclRing ring;
  int* devRingUserRanks;
  struct ncclTree tree;
  //专为CollNet优化的链式或直接通信模式
  struct ncclTree collnetChain;
  struct ncclDirect collnetDirect;
// NVLink Sharp 相关
  struct ncclNvls nvls;

  int id; // index of this channel
  uint32_t workFifoProduced; // +1 successor of last used work fifo byte // FIFO 工作队列计数器

  /* comm split sharable resources */
  struct ncclChannelPeer* collnetPeers;
  struct ncclDevChannelPeer* collnetDevPeers;
  struct ncclChannelPeer* nvlsPeers;
  struct ncclDevChannelPeer* nvlsDevPeers;
};
//管理批量的通信任务。将多个相似的通信操作（如多个 AllReduce）打包在一起处理，通过批处理减少启动开销，提高通信效率
struct ncclWorkBatchList {
  struct ncclWorkBatchList* next;
  struct ncclDevWorkBatch batch;
};

// 这个结构体通常作为基础头部，后面会跟着具体类型的数据结构，形成一个灵活的工作项描述系统。
struct alignas(16) ncclWorkList {
  struct ncclWorkList* next;
  enum ncclDevWorkType workType;// 工作类型(集合通信/点对点通信等)
  // 灵活数组成员的写法。C语言允许在结构体中定义一个灵活数组成员，它的大小是不确定的，根据实际使用情况动态分配。
  int size; // Size of struct following this node
  // ncclDevWorkColl, ncclDevWorkColLReg, ncclDevWorkP2p[]...
};
// 用于管理CollNet（Collective Network）通信句柄的链表结构
struct ncclCollnetHandleList {
  struct ncclCollnetHandleList *next;
  void* collnetHandle;
  size_t size;
  const void* buffer;
  struct ncclProxyConnector* proxyconn;
};

// NCCL 中用于管理集合通信任务的重要数据结构。像groupend这种函数其实内部就是一系列任务打包成一个group然后launch kernel。
struct ncclTaskColl {
  struct ncclTaskColl* next;
  ncclFunc_t func;// 集合通信函数指针
  void const* sendbuff;
  void* recvbuff;
  size_t count;// 要传输的元素数量
  int root;// root进程的rank（在广播等操作中使用）
  ncclDataType_t datatype;
  ncclRedOp_t opHost;// CPU端归约操作类型
  struct ncclDevRedOpFull opDev;// GPU端归约操作详细信息
  int chunkSteps, sliceSteps;// 数据分块和切片步骤数。
  // Computed later:
  size_t trafficBytes;// 预计的通信流量
  int32_t nMaxChannels:8;// 最大通道数。 这种写法叫做位域（8bit），用于在结构体中定义变量，并指定其在内存中的存储位置和大小。
  int32_t nWarps:8;// 使用的GPU warp数
  int32_t algorithm:8, protocol:8;
  uint32_t isCollnet:1, isNvls:1;
  uint32_t devFuncId:30;// 设备端函数ID
  int regBufType;
  // number of elements in planner->ipcMemQueue associated with this collective
  int nCleanupQueueElts;

  void* sendMhandle;// 发送内存句柄
  void* recvMhandle;
  void** sendNetHandles;// 发送网络句柄数组
  void** recvNetHandles;
  void** srecvNetHandles;// 发送接收网络句柄数组
  // index for IPC record lookup
  uintptr_t sendbuffOffset;
  uintptr_t recvbuffOffset;
  uintptr_t* sendbuffRmtAddrs;// 远程发送缓冲区地址
  uintptr_t* recvbuffRmtAddrs;

  // Profiler plugin
  int eActivationMask;
  void* eventHandle;
};
struct ncclTaskP2p {
  struct ncclTaskP2p* next;
  ncclFunc_t func;
  void* buff;
  size_t count;
  ncclDataType_t datatype;
  int root;// 对端进程的 rank
  size_t bytes;

  // Profiler plugin
  int eActivationMask;
  void* eventHandle;
};

struct ncclKernelPlan {
  // A kernel plan is also a callback that reclaims itself. Hence this must
  // be the first member.
  struct ncclCommCallback reclaimer; // 用于资源回收的回调

  struct ncclComm* comm;
  struct ncclKernelPlan* next;

  bool persistent; // aka captured in a graph // 是否是持久化计划（在CUDA图中捕获）
  bool isHostCbEnq;
  enum ncclDevWorkStorageType workStorageType;
  bool kernelSpecialized;
  void *kernelFn;//CUDA kernel函数指针
  struct ncclDevKernelArgs* kernelArgs;
  size_t kernelArgsSize;
  uint64_t channelMask; // bitset of which channels are present
  bool hasProxyOps; // does any channel have a non-empty proxyOpQueue
  int threadPerBlock;

  int collOpCount; // Number of collectives in this plan.
  int nWorkBatches; // Number of work batches.
  size_t workBytes; // Sum size of all work (in the fifo) in bytes.
  struct ncclIntruQueue<struct ncclWorkList, &ncclWorkList::next> workQueue;
  struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next> cleanupQueue;
  void* workBufPersistent;

  struct ncclIntruQueue<struct ncclTaskP2p, &ncclTaskP2p::next> p2pTaskQueue;
  struct ncclIntruQueue<struct ncclTaskColl, &ncclTaskColl::next> collTaskQueue;
  struct ncclIntruQueue<struct ncclProxyOp, &ncclProxyOp::enqNext> proxyOpQueue;

  // Profiler plugin
  void* groupEventHandle;
};

////////////////////////////////////////////////////////////////////////////////
// Roughly sorts ncclTaskColl's by their size descending. This structure is
// self-referential, meaning that pointers it contains internally may point
// into the structure itself. This means that it is NOT memcpy-moveable:
// 分桶策略排序。
struct ncclTaskCollSorter {
  static constexpr int UnitLog2 = 10; // 1K // 1KB 为基本单位
  static constexpr size_t UnitSize = 1<<UnitLog2;
  static constexpr int MaxLog2 = 30; // 1GB
  static constexpr size_t MaxSize = 1ull<<MaxLog2;
  // Number of bins between powers of 2. For 4 bins, the worst case out-of-order
  // relative magnitude is (5/4)-1 = 25%
  static constexpr int BitsPerPow2 = 2;
  static constexpr int BinsPerPow2 = 1<<BitsPerPow2;
  static constexpr int BinCount = 1 + (MaxLog2-UnitLog2)*BinsPerPow2;

  struct ncclTaskColl* head;
  struct ncclTaskColl* tail;
  // Least bin such that it and all above are empty.
  int binEdge;//标记了一个边界位置，这个位置及其之上的所有桶都是空的
  // Pointer to the pointer to this bin's head node which is either the
  // previous node's `next` field or `head`.
  // 所有的任务最终都会被连接成一个单向链表。
  struct ncclTaskColl** bins[BinCount];//桶数组
};

inline void ncclTaskCollSorterInsert(
    struct ncclTaskCollSorter* me, struct ncclTaskColl* x, size_t size
  ) {
  constexpr int UnitLog2 = ncclTaskCollSorter::UnitLog2;
  constexpr size_t MaxSize = ncclTaskCollSorter::MaxSize;
  constexpr int BitsPerPow2 = ncclTaskCollSorter::BitsPerPow2;
  constexpr int BinCount = ncclTaskCollSorter::BinCount;
  int bin = u32fpEncode(std::min(MaxSize, size)>>UnitLog2, BitsPerPow2);
  bin = BinCount-1 - bin; // descending bin

  if (me->bins[bin] == nullptr) {
    if (me->binEdge <= bin) {
      me->binEdge = bin+1;
      me->bins[bin] = me->tail ? &me->tail->next : &me->head;
      me->tail = x;
    } else {
      // Find successor non-empty bin after this one.
      int succ = bin+1;
      while (me->bins[succ] == nullptr) succ++;
      // What was our successor's head's previous is now our head's previous.
      me->bins[bin] = me->bins[succ];
      // The first node we insert is our tail, so that becomes our successor's
      // head's new previous.
      me->bins[succ] = &x->next;
    }
  }
  // Push a new head for this bin.
  x->next = *me->bins[bin];
  *me->bins[bin] = x;
}

inline bool ncclTaskCollSorterEmpty(struct ncclTaskCollSorter* me) {
  return me->head == nullptr;
}

// Reset sorter and return sorted linked list of its coll tasks.
inline struct ncclTaskColl* ncclTaskCollSorterDequeueAll(struct ncclTaskCollSorter* me) {
  struct ncclTaskColl* head = me->head;
  if (head != nullptr) memset(me, 0, sizeof(*me));// 重置排序器。
  return head;
}

////////////////////////////////////////////////////////////////////////////////

struct ncclCudaStreamList {
  struct ncclCudaStreamList *next;
  cudaStream_t stream;
};
//用于管理和调度 NCCL 的通信任务。
struct ncclKernelPlanner {
  //////////////////////////////////////////////////////////////////////////////
  // State for accumulating tasks between ncclGroupStart/End()
  //////////////////////////////////////////////////////////////////////////////

  struct Peer {
    bool sendSeen, recvSeen;
    struct ncclIntruQueue<struct ncclTaskP2p, &ncclTaskP2p::next> sendQueue;
    struct ncclIntruQueue<struct ncclTaskP2p, &ncclTaskP2p::next> recvQueue;
  };
  //使用排序器管理集合通信任务
  struct ncclTaskCollSorter collSorter;
  struct Peer* peers/*[nRanks]*/;
  int nTasksColl, nTasksP2p;
  bool persistent;

  // The list of user streams aggregated over all tasks present.
  struct ncclCudaStreamList* streams;
  // The most recent user stream. Ignored if streams==nullptr
  cudaStream_t streamRecent;
  // The graph capturing all user streams or invalid if none. Thus we restrict the
  // user that all streams must be captured in the same graph or not captured
  // at all. Technically we could probably relax this, but that would mean
  // collecting a different `ncclTasks` per graph and one for non-graph.
  /*
    ​一致性：NCCL需要确保所有通信任务在同一个执行上下文中运行，以避免竞争条件或未定义行为。
    ​简化设计：将所有流捕获到同一个图中，可以避免复杂的任务管理和同步逻辑。
    ​性能优化：通过统一捕获，NCCL可以更好地优化通信任务的调度和执行。
  */
  struct ncclCudaGraph capturingGraph;

  //////////////////////////////////////////////////////////////////////////////
  // Lists of tasks to be assembled into plans.
  //////////////////////////////////////////////////////////////////////////////

  struct ncclIntruQueue<struct ncclTaskColl, &ncclTaskColl::next> collTaskQueue;
  struct ncclIntruQueue<struct ncclWorkList, &ncclWorkList::next> collWorkQueue;
  struct ncclIntruQueue<struct ncclWorkList, &ncclWorkList::next> tmpCollWorkQueue;
  struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next> collCleanupQueue;

  //////////////////////////////////////////////////////////////////////////////
  // State for building current (Work-In-Progress) plan:
  //////////////////////////////////////////////////////////////////////////////
  //Work-In-Progress.管理正在构建的通信计划、为每个通道维护批处理状态、跟踪代理操作。
  struct WipPlan {
    struct Channel {
      struct {
        int workBytes; // Sum size of work metadata referenced by this batch.
        int nP2ps; // Number of p2p works in this batch
        int p2pRounds[NCCL_MAX_DEV_WORK_P2P_PER_BATCH]; // which rounds are present in this batch.
      } wipBatch; // work-in-progress batch which will be next tail of workBatchQueue
      int nWorkBatchesP2p; // number of p2p batches for this channel.
      struct ncclIntruQueue<struct ncclWorkBatchList, &ncclWorkBatchList::next> workBatchQueue;
      struct ncclIntruQueue<struct ncclProxyOp, &ncclProxyOp::enqNext> proxyOpQueue;
    } channels[MAXCHANNELS];
  } wipPlan;

  //////////////////////////////////////////////////////////////////////////////
  // State for launching built plans:
  //////////////////////////////////////////////////////////////////////////////

  // List of kernel plans built form tasks. 存储已构建的内核计划
  struct ncclIntruQueue<struct ncclKernelPlan, &ncclKernelPlan::next> planQueue;
  // First of the unlaunched kernels in `planQueue` 跟踪未启动的内核
  struct ncclKernelPlan* unlaunchedPlansHead;
};

#define NCCL_MAGIC 0x0280028002800280 // Nickel atomic number is 28.
// 每个GPU知道我要和哪些GPU通信，并且能够进行集合通信。
struct ncclComm {
  uint64_t startMagic;
  struct ncclMemoryStack memPermanent, memScoped; // 管理永久和作用域内存
  // List of destructors to run when comm is destructed
  struct ncclDestructor* destructorHead;

  struct ncclSharedResources* sharedRes;
  /* map to top parent ranks. */
  int* topParentRanks;
  int* topParentLocalRanks;
  struct ncclChannel channels[MAXCHANNELS];
  struct ncclPeerInfo* peerInfo;
  struct ncclTopoSystem* topo;
  struct ncclProxyConnector* gproxyConn;
  struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next> legacyRegCleanupQueue;

  int netPluginLoaded;
  ncclNet_t* ncclNet;
  ncclNetDeviceType netDeviceType;
  ncclCollNet_t* ncclCollNet;
  void* bootstrap;
  // Bitmasks for ncclTransportP2pSetup
  uint64_t* connectSend;
  uint64_t* connectRecv;
  struct ncclTopoGraph graphs[NCCL_NUM_ALGORITHMS];
  bool initAlgoChannels[NCCL_NUM_ALGORITHMS];
  bool runtimeConn; // if dynamic connection is supported
  bool directMode;
  int cuMemSupport;

  uint64_t magic; // Magic number for all network communication. Not a security key -- only goal is to detect mismatches.

  const char* commName;
  uint64_t commHash;
  int rank;    // my rank in the communicator
  int nRanks;  // number of GPUs in communicator
  int cudaDev; // my cuda device index
  int nvmlDev; // my nvml device index
  int compCap; // compute capability of the GPU
  int minCompCap, maxCompCap; // min/max compute capability in the communicator
  int64_t busId;   // my PCI bus ID in int format
  cpu_set_t cpuAffinity; // CPU affinity of the GPU
  int cudaArch; // matches __CUDA_ARCH__ of device

  int cpuArch;   // architecture - As defined in src/include/graph.h, e.g. x86/arm/ppc/mixed
  int cpuVendor; // vendor - As defined in src/include/graph.h

  int node;
  int nNodes;
  int localRank;
  int localRanks;
  int maxLocalRanks;
  int* rankToNode;
  int* rankToLocalRank;
  int* localRankToRank;
  // localRanks and localRanktoRank for all nodes
  struct ncclNodeRanks* nodeRanks;
  // MNNVL: Multi-Node NVLink
  int MNNVL; // true when MNNVL is available
  struct cliqueInfo clique; // Our MNNVL clique information
  int cliqueRank; // Our rank within the MNNVL clique

  bool checkPointers;
  bool dmaBufSupport;

  // Counter for tracking CUDA launches (P2P and collectives included)
  uint64_t opCount;
  // Collective operation counter
  uint64_t collOpCount;

  // Channels for collectives
  int nChannels; // connection nChannels
  int collChannels; // enqueue nChannels
  int nvlsChannels; // enqueue nChannels
  // all nvls heads stored to check if we can splitShare
  int nvlsHeads[MAXCHANNELS];
  // Channels (per peer) for p2p
  int p2pnChannels;
  int p2pnChannelsPerPeer;

  // Should this comm allocate LL buffers for network P2P connections?
  bool allocP2pNetLLBuffers;

  // Buffer sizes
  int buffSizes[NCCL_NUM_PROTOCOLS];
  int p2pChunkSize;
  int nvlsChunkSize;

  // Algorithm/Protocols thresholds
  ssize_t threadThresholds[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  float latencies[NCCL_NUM_FUNCTIONS][NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  float bandwidths[NCCL_NUM_FUNCTIONS][NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  int maxThreads[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];

  /* This attribute can indicate the states of communicators and return code of
   * asynchronous NCCL operations. */
  ncclResult_t asyncResult;

  // Flag to ask NCCL kernels to abort
  uint32_t* abortFlag;
  uint32_t* abortFlagDev;
  int* abortFlagRefCount;
  uint32_t* childAbortFlag;
  uint32_t* childAbortFlagDev;
  uint32_t destroyFlag;

  // Device side of the communicator (for cudaFree's)
  struct ncclDevComm* devComm; // actually = &ncclDevCommAndChannels::comm

  uint32_t workArgsBytes; // max size of kernel args
  uint32_t workFifoBytes; // size of workFifoBuf, power of 2
  void* workFifoBuf;
  void* workFifoBufDev;
  void* workFifoBufGdrHandle;

  // Monotonic number of bytes (mod 1<<32) consumed per channel. In cudaHost memory.
  uint32_t* workFifoConsumed/*[MAXCHANNELS]*/;
  // Last observed value of: min(workFifoConsumed[c] for c < MAXCHANNELS)
  uint32_t workFifoConsumedLeast;
  // Monotonic number of bytes (mod 1<<32) sent to fifo.
  uint32_t workFifoProduced;

  // Intra-process sync
  struct ncclComm* intraComm0; // leader of intra-process comms (self possible)
  struct ncclComm* intraNext; // next of intra-process comms, intraComm0 is head
  int intraRank;
  int intraRanks;
  uint32_t intraBarrierPhase;
  char intraPad1[64 - sizeof(uint64_t)];
  uint64_t intraBarrierCounter; // only used if this is intraComm0
  char intraPad2[64 - sizeof(uint64_t)];
  uint64_t intraBarrierGate; // only used if this is intraComm0

  struct ncclProxyState* proxyState;
  int proxyRefCountOld; /* store proxy post-atomic-sub refcount */
  // Whether this communicator uses collNet
  int collNetSupport;
  bool isOneRPN;
  uint8_t collNetSupportMatrix[4/*sum,prod,max,min*/][ncclNumTypes];
  bool intraNodeP2pSupport;
  int* collNetHeads;
  int collNetHeadsNum;
  int* collNetDenseToUserRank;
  int* collNetUserToDenseRank;
  /* sharable collNet proxy progress resource. */
  struct ncclCollNetSharedRes* collNetSharedRes;

  // NVLink SHARP (NVLS) support
  int nvlsSupport;
  int nvlsRegSupport;
  /* sharable NVLS resource. */
  struct ncclNvlsSharedRes* nvlsResources;

  // pools backed by comm->memPermanent
  struct ncclMemoryPool memPool_ncclTaskColl;
  struct ncclMemoryPool memPool_ncclTaskP2p;
  struct ncclMemoryPool memPool_ncclProxyOp;
  struct ncclMemoryPool memPool_ncclKernelPlan;

  // Next comm in this thread's active ncclGroup[Start|End](). Holds "0x1" when
  // this comm is not yet in a group.
  struct ncclComm* groupNext;
  // Subset of those in groupNext list. Holds 0x1 if not needing preconnect.
  struct ncclComm* preconnectNext;
  int persistentRefs; // number of persistent plan-lists capturing this comm
  int noncapturedRefs; // number of non-captured hostStreamPlanCallback on the stream
  struct P2pSchedulePair { int sendRank; int recvRank; } *p2pSchedule;

  struct ncclKernelPlanner planner;

  cudaMemPool_t memPool;
  // Queue of events and associated callbacks for cleaning up asynchronous work.
  // Using this is preferable to using CUDA host callbacks because host callbacks
  // won't allow the work following the callback to run until the callback completes,
  // which comes at expense to perf.
  struct ncclIntruQueue<struct ncclCommEventCallback, &ncclCommEventCallback::next> eventCallbackQueue;

  // user-created reduction ops
  int userRedOpCapacity, userRedOpFreeHead;
  ncclUserRedOp *userRedOps;

  // Queue of things for the main thread to do
  struct ncclIntruQueueMpsc<struct ncclCommCallback, &ncclCommCallback::next> callbackQueue;

  ncclConfig_t config;
  // initState is to more conveniently reclaim resources when errors happen.
  ncclResult_t initState;
  // flag to indicate if ncclCommFinalize() is called
  bool finalizeCalled;
  // shared structures for finalization
  int finalizeRankCnt;
  // group job to support multi-thread FT
  struct ncclGroupJob *groupJob;

  // Tuning plugin
  int tunerPluginLoaded;
  ncclTuner_t* tuner;
  void *tunerContext;

  // Profiler plugin
  void* profilerContext;
  uint64_t seqNumber[NCCL_NUM_FUNCTIONS];

  // buffer registration cache
  struct ncclRegCache regCache;
  int isAllNvlink;
  bool useNetPXN;
  bool useGdr;
  int splitCount;
  uint64_t endMagic;
};

static_assert(offsetof(struct ncclComm, startMagic) == 0, "startMagic must be the first field of ncclComm");
static_assert(offsetof(struct ncclComm, endMagic) == sizeof(struct ncclComm) - sizeof(uint64_t), "endMagic must be the last field of ncclComm");

enum ncclLaunchMode {
  ncclLaunchModeInvalid=0,
  ncclLaunchModeParallel,
  ncclLaunchModeGroup
};
extern enum ncclLaunchMode ncclParamLaunchMode;
//主要作用是将不同类型的内存释放请求推送到通信器的清理队列中
void ncclCommPushFree(struct ncclComm* comm, void* buf);
void ncclCommPushCudaFree(struct ncclComm* comm, void* buf);
void ncclCommPushCudaHostFree(struct ncclComm* comm, void* buf);
void ncclCommPushCudaGdrFree(struct ncclComm* comm, void* handle);
//回调轮询机制
inline ncclResult_t ncclCommPollCallbacks(struct ncclComm* comm, bool waitSome) {
  ncclResult_t result = ncclSuccess;
  //MPSC队列（Multiple Producers Single Consumer）
  struct ncclCommCallback* cb = ncclIntruQueueMpscDequeueAll(&comm->callbackQueue, waitSome);
  while (cb != nullptr) {
    struct ncclCommCallback* next = cb->next;
    ncclResult_t res1 = cb->fn(comm, cb); // may reclaim memory of cb
    if (res1 != ncclSuccess) result = res1;
    cb = next;
  }
  NCCLCHECK(result);
  return ncclSuccess;
}
//异步处理 CUDA 事件
inline ncclResult_t ncclCommPollEventCallbacks(struct ncclComm *comm) {
  ncclResult_t result = ncclSuccess;
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  while (true) {
    struct ncclCommEventCallback* cb = ncclIntruQueueHead(&comm->eventCallbackQueue);
    if (cb == nullptr) break;
    cudaError_t ok = cudaEventSynchronize(cb->event);
    if (ok == cudaErrorNotReady) break;
    ncclIntruQueueDequeue(&comm->eventCallbackQueue);
    if (ok == cudaSuccess) {
      NCCLCHECKGOTO(cb->fn(comm, cb), result, finish);
    } else {
      CUDACHECKGOTO(ok, result, finish);
    }
  }
finish:
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  return ncclSuccess;
}
//进程内同步屏障（Intra-Process Barrier）机制
inline void ncclCommIntraBarrierIn(struct ncclComm* comm, uint32_t x) {
  int phase = comm->intraBarrierPhase;
  if (comm->intraRanks == 1) {
    // Release everyone (just me).
    comm->intraBarrierGate = (uint64_t(x)<<32) | (phase^1);
  } else {
    struct ncclComm* comm0 = comm->intraComm0;
    uint64_t count = __atomic_add_fetch(&comm0->intraBarrierCounter, (uint64_t(x)<<32) + 1, __ATOMIC_RELEASE);
    if (uint32_t(count) == uint32_t(comm->intraRanks)) {
      // Reset.
      __atomic_store_n(&comm0->intraBarrierCounter, 0, __ATOMIC_RELAXED);
      // Release everyone.
      __atomic_store_n(&comm0->intraBarrierGate, (count>>32<<32) | (phase^1), __ATOMIC_RELEASE);
    }
  }
}

// returns sum of x values contributed to ncclCommIntraBarrierIn(comm, x) 
//等待机制
inline uint32_t ncclCommIntraBarrierOut(struct ncclComm* comm) {
  struct ncclComm* comm0 = comm->intraComm0;
  comm->intraBarrierPhase ^= 1;
  uint32_t phase = comm->intraBarrierPhase;
  uint64_t gate = __atomic_load_n(&comm0->intraBarrierGate, __ATOMIC_RELAXED);
  if ((gate & 1) != phase) {
    uint64_t t0 = clockNano();
    do {
      // Spin vigorously for first 5us.
      if (clockNano()-t0 >= 5*1000) sched_yield();// 让出 CPU
      gate = __atomic_load_n(&comm0->intraBarrierGate, __ATOMIC_RELAXED);
    } while ((gate & 1) != phase);
  }
  if (comm->intraRanks != 1) __atomic_thread_fence(__ATOMIC_ACQUIRE);
  return gate>>32;
}

// Scrambles the bits of non-builtin values of ncclRedOp_t according to the
// communicator memory address. Used to catch bugs so that integer handles
// associated with this communicator won't collide with handles of other
// communicatrs. This function is its own inverse.这个函数是自反的，即 Mangle(Mangle(x)) = x
//实现了一个用户自定义归约操作的句柄混淆函数
// 设计可以有效防止用户在使用多个 NCCL 通信器时误用归约操作句柄
static inline ncclRedOp_t ncclUserRedOpMangle(ncclComm *comm, ncclRedOp_t op) {
  // Preserve the built-in values.
  if(int(op) < int(ncclNumOps))
    return op;
  uint64_t h = reinterpret_cast<uint64_t>(comm);
  h ^= h >> 32;
  h *= 0x9e3779b97f4a7c13u; // Knuth's 64-bit magical hash constant
  h >>= 32; // h is now an excellent 32-bit hash of the comm pointer
  h &= int(ncclMaxRedOp); // ncclMaxRedOp is a power of 2 minus 1
  int op1 = int(h) ^ int(op);
  // Since builtin values are preserved, we also have to preserve their preimage.
  return op1 < int(ncclNumOps) ? op : ncclRedOp_t(op1);
}
//通信器状态管理的关键函数
ncclResult_t ncclCommEnsureReady(ncclComm_t comm);
ncclResult_t ncclCommSetAsyncError(ncclComm_t comm, ncclResult_t nextState);

#endif
