/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_PROXY_H_
#define NCCL_PROXY_H_

#include "device.h"
#include "info.h"
#include "socket.h"
#include "ipcsocket.h"
#include "nccl_net.h"
#include <pthread.h>
#include "shmutils.h"
#include "p2p.h"
#include "collectives.h"

typedef enum : uint8_t {
  ncclPatternRing,
  ncclPatternRingTwice,
  ncclPatternPipelineFrom,
  ncclPatternPipelineTo,
  ncclPatternTreeUp,
  ncclPatternTreeDown,
  ncclPatternTreeUpDown,
  ncclPatternCollnetChain,
  ncclPatternCollnetDirect,
  ncclPatternNvls,
  ncclPatternNvlsTree,
  ncclPatternPatUp,
  ncclPatternPatDown,
  ncclPatternSend,
  ncclPatternRecv
} ncclPattern_t;

enum ncclProxyOpState { ncclProxyOpNone, ncclProxyOpReady, ncclProxyOpProgress };

struct ncclProxyArgs;
typedef ncclResult_t (*proxyProgressFunc_t)(struct ncclProxyState*, struct ncclProxyArgs*);

#define NCCL_PROXY_MAX_SUBS MAXCHANNELS
static_assert(2*NCCL_MAX_DEV_WORK_P2P_PER_BATCH <= MAXCHANNELS, "Not enough sub space for max work elements");

union ncclProxyOpSpecifics {
  struct {
    size_t sizePerRank;
    int nNodes, node;
  } collnetDirect;
};

struct ncclProxyOp {
  struct ncclProxyConnection* connection;
  ssize_t nbytes;
  uint64_t opCount;
  int root;
  int next;
  int nsteps;
  size_t chunkSize;
  size_t sliceSize;
  size_t loopSize;
  size_t loopOffset;
  size_t channelSize;
  uint8_t sliceSteps;
  uint8_t chunkSteps;
  uint8_t channelId;
  uint8_t /*ncclDataType_t*/ dtype;
  uint8_t /*ncclDevRedOp_t*/ redOp;
  uint8_t /*ncclFunc_t*/ coll;
  uint8_t /*ncclPattern_t*/ pattern;
  uint8_t protocol;
  uint8_t algorithm;
  uint8_t reg;
  // collnet/p2p/coll buffer reg handles
  void* sendMhandle;
  void* recvMhandle;
  uint8_t* sendbuff;
  uint8_t* recvbuff;
  int isOneRPN;
  RingAlgorithm *ringAlgo;
  union ncclProxyOpSpecifics specifics;

  // Profiler plugin
  union {
    struct ncclTaskColl* coll;
    struct ncclTaskP2p* p2p;
  } task;

  int eActivationMask;
  void* taskEventHandle;
  int rank;
  int peer;
  pid_t pid;
  void* profilerContext;

  struct ncclProxyOp *enqNext;
};

struct ncclProxySubArgs {
  struct ncclProxyConnection* connection;
  int reg;
  // collnet handles
  void* sendMhandle;
  void* recvMhandle;
  uint8_t* sendbuff;
  uint8_t* recvbuff;
  size_t offset;
  ssize_t loopSize;
  ssize_t loopOffset;
  int channelId;
  int nsteps;
  ssize_t nbytes;
  ssize_t chunkSize;
  int peer;
  int isOneRPN;
  RingAlgorithm *ringAlgo;
  int groupSize; // Number of consecutive sub operations sharing the same recvComm
  uint64_t base;
  uint64_t posted;
  uint64_t received;
  uint64_t flushed;
  uint64_t transmitted;
  uint64_t done;
  uint64_t end;
  int regBufferReady;
  void* requests[NCCL_STEPS];

  // Profiler plugin
  int eActivationMask;
  int rank;
  pid_t pid;
  void* profilerContext;
  void* taskEventHandle;
  void* opEventHandle;
  void* stepEventHandles[NCCL_STEPS];
  size_t transSize;

  void* recvRequestsCache[NCCL_STEPS];
  int recvRequestsSubCount;
};

struct ncclProxyArgs {
  struct ncclProxySubArgs subs[NCCL_PROXY_MAX_SUBS];
  proxyProgressFunc_t progress;
  int nsubs;
  int done;
  int onePPN;
  uint64_t opCount;
  int sliceSteps;
  int chunkSteps;
  size_t chunkSize;
  size_t totalSendSize;
  size_t totalRecvSize;
  size_t sendSizePerRound;
  size_t recvSizePerRound;
  uint8_t /*ncclDataType_t*/ dtype;
  uint8_t /*ncclDevRedOp_t*/ redOp;
  uint8_t /*ncclPattern_t*/ pattern;
  uint8_t /*ncclFunc_t*/ coll;
  uint8_t protocol;
  uint8_t algorithm;
  int state;
  char* sharedBuff[NCCL_STEPS];
  int sharedSize[NCCL_STEPS];

  int idle;

  // Element linking
  struct ncclProxyArgs* next;
  struct ncclProxyArgs* nextPeer;
  struct ncclProxyArgs** proxyAppendPtr;

  union ncclProxyOpSpecifics specifics;
};
#define NCCL_MAX_NETDEVS 128

// ProxyOps are used to communicate between main thread and service thread
// Make sure we have enough to store two full rounds of operations on all channels.
// Otherwise we'd be unable to post half of them to free new elements. Each
// p2p work contains a send and recv proxy op hence the 2x before it.
#define MAX_OPS_PER_PEER (2*MAXCHANNELS*2*NCCL_MAX_DEV_WORK_P2P_PER_BATCH)

struct ncclProxyOpsPool {
  struct ncclProxyOp ops[MAX_OPS_PER_PEER*NCCL_MAX_LOCAL_RANKS];
  volatile int nextOps;
  volatile int nextOpsEnd;
  volatile int freeOps[NCCL_MAX_LOCAL_RANKS];
  pthread_mutex_t mutex;
  pthread_cond_t cond;
};

struct ncclProxyOps {
  ncclProxyOpsPool* pool;
  ncclShmHandle_t handle;
  int count;
  int freeOp;
  int nextOps;
  int nextOpsEnd;
};

struct ncclProxySharedP2p {
  int refcount;
  int size;
  char* cudaBuff;
  char* hostBuff;
  // CUDA IPC
  ncclIpcDesc ipcDesc;
  struct ncclProxyArgs* proxyAppend[MAXCHANNELS]; // Separate send and recv
};

struct ncclProxyPeer {
  struct ncclProxySharedP2p send;
  struct ncclProxySharedP2p recv;
};

struct ncclSharedNetComms {
  void* sendComm[MAXCHANNELS];
  void* recvComm[MAXCHANNELS];
  int sendRefCount[MAXCHANNELS];
  int recvRefCount[MAXCHANNELS];
};

struct ncclProxyPool;
struct ncclProxyProgressState {
  // Used by main threads to send work to progress thread
  struct ncclProxyOpsPool* opsPool;
  ncclShmHandle_t handle;
  char opsPoolShmSuffix[6];

  pthread_t thread;
  volatile int stop;
  struct ncclProxyPeer** localPeers;
  struct ncclSharedNetComms* netComms[NCCL_MAX_NETDEVS];
  struct ncclProxyArgs* active;
  struct ncclProxyArgs* pool;
  struct ncclProxyPool* pools;
  int nextOps;
};

// Expected proxy response fifo
struct ncclExpectedProxyResponse {
  void*                             opId;
  int                               respSize;
  bool                              done;
  void*                             respBuff;
  ncclResult_t                      res;
  struct ncclExpectedProxyResponse* next;
};
//异步操作
struct ncclProxyAsyncOp {
  int type;//表示异步操作的类型，具体的值会定义不同的操作（例如，发送数据、接收数据等）。
  struct ncclProxyConnection* connection;
  int reqSize, respSize;//分别表示请求和响应的大小。这通常用于网络通信，指示需要发送多少数据和期望接收多少数据。
  char *reqBuff, *respBuff;//分别指向请求和响应的数据缓冲区。Proxy 线程会从 reqBuff 发送数据，并将接收到的数据存入 respBuff 。
  void* opId;
  ncclProxyAsyncOp* next;
};
//代表 Proxy 线程所管理的本地 Peer（对等方）的信息
struct ncclProxyLocalPeer {
  struct ncclSocket sock;
  int tpRank;
  int tpLocalRank;
  ncclProxyAsyncOp* asyncOps;
  int asyncOpCounter;
};

// Common response header for all proxyOps
// We pack this into a struct to reduce the number of blocking send and recv calls
struct ncclProxyRpcResponseHeader {
  void* opId;
  ncclResult_t res;
  int respSize;
};

// UDS support
struct ncclIpcHdr {
  int type;
  int rank;
  int reqSize;
  int respSize;
  void *opId;
  uint64_t data[16]; // 128-bytes
};

struct ncclProxyState {
  int refCount;
  int tpRank;
  int tpnRanks;
  int tpLocalnRanks;
  int cudaDev;
  int p2pnChannels;
  int p2pChunkSize;
  int nChannels;
  int buffSizes[NCCL_NUM_PROTOCOLS];
  bool allocP2pNetLLBuffers;
  bool dmaBufSupport;
  ncclNet_t* ncclNet;
  ncclCollNet_t* ncclCollNet;
  uint32_t* abortFlag;
  bool directMode;
  // Service threads
  pthread_t thread;
  pthread_t threadUDS;
  struct ncclSocket* listenSock;
  struct ncclIpcSocket ipcSock;
  int stop;
  CUcontext cudaCtx;
  ncclResult_t asyncResult;

  // Used by main thread
  union ncclSocketAddress* peerAddresses;
  struct ncclSocket* peerSocks;
  struct ncclProxyOps* proxyOps;
  void** sharedDevMems;
  struct ncclIpcSocket peerIpcSock; // cuMEM API support (UDS)
  uint64_t *peerAddressesUDS; // cuMem API support (UDS)

  // Progress thread
  struct ncclProxyProgressState progressState;

  // Profiler plugin
  void* profilerContext;

  // Queue of expected responses from the proxy
  struct ncclExpectedProxyResponse* expectedResponses;
};

enum proxyConnectState {
  connUninitialized     = 0,
  connInitialized       = 1,
  connSharedInitialized = 2,
  connSetupDone         = 3,
  connConnected         = 4,
  numConnStates         = 5
};

struct ncclProxyConnection {
  int send, transport, shared;
  int tpLocalRank, sameProcess;
  struct ncclSocket* sock;
  struct ncclTransportComm* tcomm;
  struct ncclProxyArgs *proxyAppend;
  struct ncclProxyArgs **proxyAppendPtr;
  void* transportResources;
  ncclNetDeviceHandle_t* netDeviceHandle;
  void* mhandles[NCCL_NUM_PROTOCOLS];
  proxyConnectState state;
  struct ncclCollNetSharedRes* collNet;
  int needsProxyProgress;
};

typedef ncclResult_t (*threadFunc_t)(struct ncclProxyArgs*);

enum proxyMode {
  proxyRing = 0,
  proxyFrom = 1,
  proxyTo = 2
};

ncclResult_t ncclProxySaveOp(struct ncclComm* comm, struct ncclProxyOp* proxyOp, bool *justInquire);
ncclResult_t ncclProxyStart(struct ncclComm* comm);
ncclResult_t ncclProxyInit(struct ncclComm* comm, struct ncclSocket* sock, union ncclSocketAddress* peerAddresses, uint64_t *peerAddressesUDS);
ncclResult_t ncclProxyCreate(struct ncclComm* comm);
ncclResult_t ncclProxyConnect(struct ncclComm* comm, int transport, int send, int proxyRank, struct ncclProxyConnector* proxyConn);

// NB: ncclProxyMsgTypeStr[] in proxy.cc needs to match
/*
代理支持的消息类型
*/
enum ncclProxyMsgType {
  ncclProxyMsgInit = 1,//通常用于创建新的通信上下文，初始化 proxy 的内部状态
  ncclProxyMsgSharedInit = 2,//共享初始化消息。可能用于多个进程共享某些资源时的初始化，比如共享内存或文件描述符。
  ncclProxyMsgSetup = 3,//设置阶段的消息。用于配置通信参数，准备后续的数据传输。
  ncclProxyMsgConnect = 4,//	连接请求消息。用于建立与其他节点或设备的连接。
  ncclProxyMsgStart = 5,//启动通信的消息。在所有准备工作完成后，通过该消息触发实际的数据传输。
  ncclProxyMsgClose = 6,//	关闭连接的消息。用于正常结束通信并释放相关资源。
  ncclProxyMsgAbort = 7,//异常终止消息。当发生错误或需要紧急停止通信时发送此消息。
  ncclProxyMsgStop = 8,// 停止通信的消息。与 ncclProxyMsgClose 不同，Stop 可能是更温和的停止方式。
  ncclProxyMsgGetFd = 9, // cuMem API support (UDS)获取文件描述符的消息，用于 cuMem API 支持（如 Unix Domain Socket）。
  ncclProxyMsgQueryFd = 10,//查询文件描述符的状态或相关信息。
  ncclProxyMsgRegister = 11,//注册资源的消息。通常用于将某些资源（如内存缓冲区）注册到 NCCL 中。
  ncclProxyMsgDeregister = 12//注销资源的消息。用于从 NCCL 中移除之前注册的资源，并释放相关资源。
};

// This function is called by a client of the proxy that needs to invoke any of the non-progress proxyOp types
// Call this function on the client, supplying a locally unique opId. Then, poll on the return value of
// ncclPollProxyResponse(), supplying the same opId to confirm the operation has completed
ncclResult_t ncclProxyCallAsync(struct ncclComm* comm, struct ncclProxyConnector* proxyConn, int type, void* reqBuff, int reqSize, int respSize, void* opId);

// This function will internally call ncclProxyCallAsync() and spin until ncclPollProxyResponse() confirms the result is received
ncclResult_t ncclProxyCallBlocking(struct ncclComm* comm, struct ncclProxyConnector* proxyConn, int type, void* reqBuff, int reqSize, void* respBuff, int respSize);
ncclResult_t ncclPollProxyResponse(struct ncclComm* comm, struct ncclProxyConnector* proxyConn, void* respBuff, void* opId);

// UDS support
ncclResult_t ncclProxyClientGetFdBlocking(struct ncclComm* comm, int rank, void *handle, int* convertedFd);
ncclResult_t ncclProxyClientQueryFdBlocking(struct ncclComm* comm, struct ncclProxyConnector* proxyConn, int localFd, int* rmtFd);

ncclResult_t ncclProxyStop(struct ncclComm* comm);
ncclResult_t ncclProxyShmUnlink(struct ncclComm* comm);
ncclResult_t ncclProxyDestroy(struct ncclComm* comm);
#endif
