/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "core.h"
#include "socket.h"
#include "net.h"
#include "param.h"

#include <pthread.h>
#include <stdlib.h>
#include <poll.h>
#include <limits.h>
#include <fcntl.h>

/* Init functions */
static int ncclNetIfs = -1;
struct ncclNetSocketDev {
  union ncclSocketAddress addr;
  char devName[MAX_IF_NAME_SIZE];
  char* pciPath;
};
static struct ncclNetSocketDev ncclNetSocketDevs[MAX_IFS];

pthread_mutex_t ncclNetSocketLock = PTHREAD_MUTEX_INITIALIZER;
// 获取设备的pci路径
static ncclResult_t ncclNetSocketGetPciPath(char* devName, char** pciPath) {
  char devicePath[PATH_MAX];
  snprintf(devicePath, PATH_MAX, "/sys/class/net/%s/device", devName);
  // May return NULL if the file doesn't exist.
  *pciPath = realpath(devicePath, NULL);
  return ncclSuccess;
}

ncclResult_t ncclNetSocketInit(ncclDebugLogger_t logFunction) {
  if (ncclNetIfs == -1) {
    pthread_mutex_lock(&ncclNetSocketLock);
    if (ncclNetIfs == -1) {
      char names[MAX_IF_NAME_SIZE*MAX_IFS];
      union ncclSocketAddress addrs[MAX_IFS];
      ncclNetIfs = ncclFindInterfaces(names, addrs, MAX_IF_NAME_SIZE, MAX_IFS);
      if (ncclNetIfs <= 0) {
        WARN("NET/Socket : no interface found");
        pthread_mutex_unlock(&ncclNetSocketLock);
        return ncclInternalError;
      } else {
        #define MAX_LINE_LEN (2047)
        char line[MAX_LINE_LEN+1];
        char addrline[SOCKET_NAME_MAXLEN+1];
        line[0] = '\0';
        addrline[SOCKET_NAME_MAXLEN] = '\0';
        for (int i=0; i<ncclNetIfs; i++) {
          strcpy(ncclNetSocketDevs[i].devName, names+i*MAX_IF_NAME_SIZE);
          memcpy(&ncclNetSocketDevs[i].addr, addrs+i, sizeof(union ncclSocketAddress));
          NCCLCHECK(ncclNetSocketGetPciPath(ncclNetSocketDevs[i].devName, &ncclNetSocketDevs[i].pciPath));
          snprintf(line+strlen(line), MAX_LINE_LEN-strlen(line), " [%d]%s:%s", i, names+i*MAX_IF_NAME_SIZE,
              ncclSocketToString(&addrs[i], addrline));
        }
        line[MAX_LINE_LEN] = '\0';
        INFO(NCCL_INIT|NCCL_NET,"NET/Socket : Using%s", line);
      }
    }
    pthread_mutex_unlock(&ncclNetSocketLock);
  }
  return ncclSuccess;
}

ncclResult_t ncclNetSocketDevices(int* ndev) {
  *ndev = ncclNetIfs;
  return ncclSuccess;
}

static ncclResult_t ncclNetSocketGetSpeed(char* devName, int* speed) {
  ncclResult_t ret = ncclSuccess;
  *speed = 0;
  char speedPath[PATH_MAX];
  snprintf(speedPath, sizeof(speedPath), "/sys/class/net/%s/speed", devName);
  int fd = -1;
  SYSCHECKSYNC(open(speedPath, O_RDONLY), "open", fd);
  if (fd != -1) {
    char speedStr[] = "        ";
    int n;
    // Allow this to silently fail
    n = read(fd, speedStr, sizeof(speedStr)-1);
    if (n > 0) {
      *speed = strtol(speedStr, NULL, 0);
    }
  }
  if (*speed <= 0) {
    INFO(NCCL_NET, "Could not get speed from %s. Defaulting to 10 Gbps.", speedPath);
    *speed = 10000;
  }
  if (fd != -1) SYSCHECK(close(fd), "close");
  return ret;
}

ncclResult_t ncclNetSocketGetProperties(int dev, ncclNetProperties_t* props) {
  props->name = ncclNetSocketDevs[dev].devName;
  props->pciPath = ncclNetSocketDevs[dev].pciPath;
  props->guid = dev;
  props->ptrSupport = NCCL_PTR_HOST;
  props->regIsGlobal = 0;
  props->forceFlush = 0;
  NCCLCHECK(ncclNetSocketGetSpeed(props->name, &props->speed));
  props->latency = 0; // Not set
  props->port = 0;
  props->maxComms = 65536;
  props->maxRecvs = 1;
  props->netDeviceType    = NCCL_NET_DEVICE_HOST;
  props->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;
  props->maxP2pBytes = NCCL_MAX_NET_SIZE_BYTES;
  return ncclSuccess;
}

/* Communication functions */

#define MAX_SOCKETS 64
#define MAX_THREADS 16
#define MAX_REQUESTS NCCL_NET_MAX_REQUESTS
#define MIN_CHUNKSIZE (64*1024)

NCCL_PARAM(SocketNsocksPerThread, "NSOCKS_PERTHREAD", -2);
NCCL_PARAM(SocketNthreads, "SOCKET_NTHREADS", -2);

enum ncclNetSocketCommState {
  ncclNetSocketCommStateStart = 0,// 初始状态，表示通信尚未开始
  ncclNetSocketCommStateConnect = 1,// 连接状态，表示正在尝试建立连接
  ncclNetSocketCommStateAccept = 3,// 接受状态，表示正在等待接受连接
  ncclNetSocketCommStateSend = 4,// 发送状态，表示正在发送数据
  ncclNetSocketCommStateRecv = 5,// 接收状态，表示正在接收数据
};
//这个结构体存储了通信过程中的状态信息：状态保持 ：在非阻塞操作中，一个连接可能需要多次函数调用才能完成。
// 这个结构体保存了中间状态，使得每次调用都能从上次中断的地方继续。(是实现高效非阻塞通信的关键)
struct ncclNetSocketCommStage {
  enum ncclNetSocketCommState state;// 当前通信状态
  uint8_t iteration;// 当前处理的套接字索引
  struct ncclSocket* sock;// 当前正在处理的套接字
  struct ncclNetSocketComm* comm;// 关联的通信对象
};
//用于在网络连接建立过程中传递必要的连接信息
struct ncclNetSocketHandle {
  union ncclSocketAddress connectAddr;// 连接地址，包含IP和端口信息,存储目标节点的网络地址信息。
  uint64_t magic; // random number to help debugging
  int nSocks;// 每个通信通道使用的套接字数量，多个套接字可以并行传输数据，提高带宽利用率
  int nThreads; // 处理通信的线程数量
  struct ncclNetSocketCommStage stage;  // 连接状态信息
};

/*
- 请求层 ： ncclNetSocketRequest 对应高级API（如 ncclNetSocketIsend ）调用，表示一个完整的通信操作
- 任务分解 ：大请求被分解为多个 ncclNetSocketTask ，每个任务处理一部分数据
- 任务分配 ：任务被分配到不同线程的 ncclNetSocketTaskQueue 中
- 并行执行 ：多个辅助线程并行处理各自队列中的任务
*/


//基本的通信任务，由辅助线程执行
struct ncclNetSocketTask {
  int op;// 操作类型：NCCL_SOCKET_SEND 或 NCCL_SOCKET_RECV
  void* data;
  int size;
  struct ncclSocket* sock;// 用于此任务的套接字
  int offset;// 当前已处理的数据偏移量，用于跟踪进度
  int used; // 使用标志，1表示任务正在使用中，0表示可用
  ncclResult_t result; // 任务执行结果
};
/*
表示一个完整的通信请求，可能会被分解为多个任务。一个请求代表一个完整的发送或接收操作，它会：
1. 首先通过控制套接字交换消息大小信息
2. 然后将大消息分割成多个较小的块
3. 为每个块创建一个任务，分配给不同的套接字和线程处理

*/
struct ncclNetSocketRequest {
  int op;// 操作类型：NCCL_SOCKET_SEND 或 NCCL_SOCKET_RECV
  void* data;
  int size;
  struct ncclSocket* ctrlSock; // 控制套接字，用于传输元数据（如消息大小）
  int offset;// 当前处理进度
  int used;// 使用状态：0=未使用，1=初始化中，2=数据传输中
  struct ncclNetSocketComm* comm;// 所属的通信对象
  struct ncclNetSocketTask* tasks[MAX_SOCKETS];// 子任务数组，请求被分解成的多个任务
  int nSubs; // 子任务数量
};

struct ncclNetSocketTaskQueue {
  int next;// 下一个可用任务槽的索引
  int len;// 队列长度（任务槽总数）
  struct ncclNetSocketTask* tasks;// 任务数组
};
//管理每个辅助线程的资源和状态
struct ncclNetSocketThreadResources {
  struct ncclNetSocketTaskQueue threadTaskQueue;// 线程专用的任务队列，存储该线程需要处理的通信任务
  int stop;  // 控制线程终止的标志位，当设置为1时，线程会退出循环并结束执行
  struct ncclNetSocketComm* comm;// 指向所属通信对象的指针,使线程能够访问通信相关的资源
  pthread_mutex_t threadLock;//访问任务队列时使用
  pthread_cond_t  threadCond;//用于主线程通知辅助线程有新任务，或者让辅助线程在空闲时等待
};
//表示一个监听通信对象
struct ncclNetSocketListenComm {
  struct ncclSocket sock; // 监听套接字，用于接受连接请求
  struct ncclNetSocketCommStage stage;// 连接状态信息，用于非阻塞连接处理
  int nSocks;// 每个通信通道使用的并行套接字数量
  int nThreads;// 处理通信的线程数量
  int dev;// 关联的网络设备ID
};
//表示一个完整的通信对象，包含发送或接收数据所需的所有资源
struct ncclNetSocketComm {
  struct ncclSocket ctrlSock;// 控制套接字，用于传输元数据
  struct ncclSocket socks[MAX_SOCKETS];// 数据套接字数组，用于并行传输数据
  int dev;
  int cudaDev;
  int nSocks; // 实际使用的套接字数量
  int nThreads; // 实际使用的线程数量
  int nextSock;// 下一个要使用的套接字索引，用于轮询分配
  struct ncclNetSocketRequest requests[MAX_REQUESTS]; // 请求数组，存储活跃的通信请求
  pthread_t helperThread[MAX_THREADS]; // 辅助线程数组,这些线程负责实际的数据传输.
  struct ncclNetSocketThreadResources threadResources[MAX_THREADS];//线程资源数组，每个线程对应一个资源结构
};

void* persistentSocketThread(void *args_) {
  struct ncclNetSocketThreadResources* resource = (struct ncclNetSocketThreadResources*)args_;
  struct ncclNetSocketComm* comm = resource->comm;
  struct ncclNetSocketTaskQueue* myQueue = &resource->threadTaskQueue;
  int nSocksPerThread = comm->nSocks / comm->nThreads;
  while (1) {
    int idle = 1;
    int mark = myQueue->next; // mark newest task seen
    for (int i=0; i<myQueue->len; i+=nSocksPerThread) {//这里的理解是处理的任务数量，不是socket数量。
      int repeat;
      do {
        repeat = 0;
        for (int j=0; j<nSocksPerThread; j++) {
          struct ncclNetSocketTask* r = myQueue->tasks+i+j;
          if (r != NULL && r->used == 1 && r->offset < r->size) {
            r->result = ncclSocketProgress(r->op, r->sock, r->data, r->size, &r->offset);//非阻塞
            if (r->result != ncclSuccess) {
              WARN("NET/Socket : socket progress error");
              return NULL;
            }
            idle = 0;
            //如果某个任务尚未完成（offset < size），则设置 repeat=1，继续重试这一批任务。
            if (r->offset < r->size) repeat = 1;
          }
        }
      } while (repeat);
    }
    if (idle) {//当前没有任务要处理时（idle == 1），线程进入等待状态。
      pthread_mutex_lock(&resource->threadLock);
      while (mark == myQueue->next && resource->stop == 0) { // no new tasks, wait
        pthread_cond_wait(&resource->threadCond, &resource->threadLock);
      }
      pthread_mutex_unlock(&resource->threadLock);
    }
    if (resource->stop) return NULL;//没有外部信号是不会终止的，这通常用于后台服务线程。
  }
}

ncclResult_t ncclNetSocketGetNsockNthread(int dev, int* ns, int* nt) {
  ncclResult_t ret = ncclSuccess;
  int nSocksPerThread = ncclParamSocketNsocksPerThread();
  int nThreads = ncclParamSocketNthreads();
  if (nThreads > MAX_THREADS) {
    WARN("NET/Socket : NCCL_SOCKET_NTHREADS is greater than the maximum allowed, setting to %d", MAX_THREADS);
    nThreads = MAX_THREADS;
  }
  int fd = -1;
  int nSocks;
  if (nThreads == -2 || nSocksPerThread == -2) {
    // Auto-detection
    int autoNt=0, autoNs=1; // By default, we only use the main thread and do not spawn extra threads
    char vendorPath[PATH_MAX];
    snprintf(vendorPath, PATH_MAX, "/sys/class/net/%s/device/vendor", ncclNetSocketDevs[dev].devName);
    // Coverity is wrong.  NULL second argument to realpath() is OK by POSIX.1-2008.
    // coverity[alias_transfer:FALSE]
    char* rPath = realpath(vendorPath, NULL);//打开对应的网络设备的vendor信息
    fd = open(rPath, O_RDONLY);
    free(rPath);
    if (fd == -1) {
      // Could not find device vendor. This is handled silently so
      // we don't want to print an INFO error.
      TRACE(NCCL_NET, "Open of %s failed : %s", vendorPath, strerror(errno));
      goto end;
    }
    char vendor[7];
    strncpy(vendor, "0x0000", 7);
    SYSCHECKGOTO(read(fd, vendor, 6), "read", ret, fail);
    if (strcmp(vendor, "0x1d0f") == 0) { // AWS
      autoNt = 2;
      autoNs = 8;
    } else if (strcmp(vendor, "0x1ae0") == 0) { // GCP （Google的云平台）
      autoNt = 4;
      autoNs = 1;
    }
end:
    if (nThreads == -2) nThreads = autoNt;
    if (nSocksPerThread == -2) nSocksPerThread = autoNs;
  }
  nSocks = nSocksPerThread * nThreads;
  if (nSocks > MAX_SOCKETS) {
    nSocksPerThread = MAX_SOCKETS/nThreads;
    WARN("NET/Socket : the total number of sockets is greater than the maximum allowed, setting NCCL_NSOCKS_PERTHREAD to %d", nSocksPerThread);
    nSocks = nSocksPerThread * nThreads;
  }
  *ns = nSocks;
  *nt = nThreads;
  if (nSocks > 0) INFO(NCCL_INIT, "NET/Socket: Using %d threads and %d sockets per thread", nThreads, nSocksPerThread);
exit:
  if (fd != -1) close(fd);
  return ret;
fail:
  goto exit;
}
//根据指定的网络设备dev号创建并初始化一个监听通信对象。这里listen的是本地的网络接口
ncclResult_t ncclNetSocketListen(int dev, void* opaqueHandle, void** listenComm) {
  if (dev < 0 || dev >= ncclNetIfs) { // data transfer socket is based on specified dev
    WARN("NET/Socket : ncclNetSocketListen dev=%d ncclNetIfs=%d", dev, ncclNetIfs);
    return ncclInternalError;
  }
  ncclResult_t ret = ncclSuccess;
  struct ncclNetSocketHandle* handle = (struct ncclNetSocketHandle*) opaqueHandle;
  memset(handle, 0, sizeof(struct ncclNetSocketHandle));
  static_assert(sizeof(struct ncclNetSocketHandle) <= NCCL_NET_HANDLE_MAXSIZE, "ncclNetSocketHandle size too large");
  struct ncclNetSocketListenComm* comm;
  NCCLCHECK(ncclCalloc(&comm, 1));
  handle->magic = NCCL_SOCKET_MAGIC;
  NCCLCHECKGOTO(ncclSocketInit(&comm->sock, &ncclNetSocketDevs[dev].addr, handle->magic, ncclSocketTypeNetSocket, NULL, 1), ret, fail);
  NCCLCHECKGOTO(ncclSocketListen(&comm->sock), ret, fail);
  NCCLCHECKGOTO(ncclSocketGetAddr(&comm->sock, &handle->connectAddr), ret, fail);
  NCCLCHECKGOTO(ncclNetSocketGetNsockNthread(dev, &comm->nSocks, &comm->nThreads), ret, fail);
  handle->nSocks = comm->nSocks;
  handle->nThreads = comm->nThreads;
  comm->dev = dev;
  *listenComm = comm;
exit:
  return ret;
fail:
  (void)ncclSocketClose(&comm->sock);
  free(comm);
  goto exit;
}
//非阻塞式的连接建立过程，允许NCCL在高并发环境中高效地建立多个连接。opaqueHandle : 包含远程节点连接信息的句柄,sendDevComm : 设备卸载相关参数（Socket后端不使用）
//sendComm 封装了发送端所需的所有资源和状态。（其实就是connect到远端的各种信息，之后可以用这个comm去发送信息。）
ncclResult_t ncclNetSocketConnect(int dev, void* opaqueHandle, void** sendComm, ncclNetDeviceHandle_t** /*sendDevComm*/) {
  if (dev < 0 || dev >= ncclNetIfs) { // data transfer socket is based on specified dev
    return ncclInternalError;
  }

  int ready;
  /*
  - 从句柄中提取连接状态信息
  - 根据当前状态决定跳转到哪个阶段
  - 如果是首次调用，分配通信对象并初始化
  */
  struct ncclNetSocketHandle* handle = (struct ncclNetSocketHandle*) opaqueHandle;
  struct ncclNetSocketCommStage* stage = &handle->stage;
  struct ncclNetSocketComm* comm = stage->comm;
  uint8_t i = stage->iteration;
  struct ncclSocket* sock = stage->sock;
  *sendComm = NULL;

  if (stage->state == ncclNetSocketCommStateConnect) goto socket_connect_check;
  if (stage->state == ncclNetSocketCommStateSend) goto socket_send;
  //如果是首次调用，分配通信对象并初始化
  NCCLCHECK(ncclCalloc(&comm, 1));
  stage->comm = comm;
  comm->nSocks = handle->nSocks;
  comm->nThreads = handle->nThreads;
  comm->dev = dev;
  CUDACHECK(cudaGetDevice(&comm->cudaDev));
  /*
    - 为每个数据通道和控制通道创建套接字
    - 初始化套接字并尝试连接
    - 检查连接是否就绪，如果未就绪则返回（非阻塞）
    - 连接就绪后，更新状态为发送阶段
  */
  for (; i<comm->nSocks+1; i++) {//这里nSocks+1是因为还要建立一个控制socket连接。
    sock = (i == comm->nSocks) ? &comm->ctrlSock : comm->socks+i;
    //每个socket都是连接目标地址的
    NCCLCHECK(ncclSocketInit(sock, &handle->connectAddr, handle->magic, ncclSocketTypeNetSocket, NULL, 1));

    stage->sock = sock;
    stage->state = ncclNetSocketCommStateConnect;//连接中
    stage->iteration = i;//socket索引
    NCCLCHECK(ncclSocketConnect(sock));

socket_connect_check:
    NCCLCHECK(ncclSocketReady(sock, &ready));
    //如果没有ready，表示连接还没有完成，这里不会阻塞等待，而是返回ncclSuccess，表示可以继续处理其他任务。（注意上面对i的记录很关键，决定着下次调用从哪个socket开始）
    if (! ready) return ncclSuccess;//这是NCCL中实现非阻塞连接的关键机制。由于TCP连接建立需要时间（三次握手过程），NCCL不会阻塞等待连接完成，而是通过这种方式实现异步连接
    stage->state = ncclNetSocketCommStateSend;//连接成功，准备发送数据。

socket_send:
/*
    - 向远程节点发送套接字索引（i）
  - 如果发送未完成，返回（非阻塞）
  - 所有套接字都连接并发送索引后，设置输出参数并返回
  //这里发送变量 i 的目的是为了告知接收端当前套接字的用途和身份- 当 i == nSocks 时，表示这是控制套接字,当 i < nSocks 时，表示这是第 i 个数据套接字
  接收端根据接收到的索引，接收端可以正确地将套接字映射到对应的位置。
*/
    int done = 0;
    NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_SEND, sock, &i, sizeof(uint8_t), &done));
    if (done == 0) return ncclSuccess;
  }
  *sendComm = comm;
  return ncclSuccess;
}

ncclResult_t ncclNetSocketAccept(void* listenComm, void** recvComm, ncclNetDeviceHandle_t** /*recvDevComm*/) {
  struct ncclNetSocketListenComm* lComm = (struct ncclNetSocketListenComm*)listenComm;
  struct ncclNetSocketCommStage* stage = &lComm->stage;
  struct ncclNetSocketComm* rComm = stage->comm;
  uint8_t i = stage->iteration;
  struct ncclSocket* sock = stage->sock;
  int ready;

  *recvComm = NULL;
  if (stage->state == ncclNetSocketCommStateAccept) goto socket_accept_check;
  if (stage->state == ncclNetSocketCommStateRecv) goto socket_recv;

  NCCLCHECK(ncclCalloc(&rComm, 1));
  stage->comm = rComm;
  rComm->nSocks = lComm->nSocks;
  rComm->nThreads = lComm->nThreads;
  rComm->dev = lComm->dev;
  CUDACHECK(cudaGetDevice(&rComm->cudaDev));
  for (; i<rComm->nSocks+1; i++) {
    uint8_t sendSockIdx;

    NCCLCHECK(ncclCalloc(&sock, 1));
    NCCLCHECK(ncclSocketInit(sock));
    stage->sock = sock;
    stage->state = ncclNetSocketCommStateAccept;
    stage->iteration = i;
    NCCLCHECK(ncclSocketAccept(sock, &lComm->sock));

socket_accept_check:
    NCCLCHECK(ncclSocketReady(sock, &ready));
    if (!ready) return ncclSuccess;

    stage->state = ncclNetSocketCommStateRecv;
socket_recv:
    int done = 0;
    NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_RECV, sock, &sendSockIdx, sizeof(uint8_t), &done));
    if (done == 0) return ncclSuccess;

    if (sendSockIdx == rComm->nSocks)
      memcpy(&rComm->ctrlSock, sock, sizeof(struct ncclSocket));
    else
      memcpy(rComm->socks+sendSockIdx, sock, sizeof(struct ncclSocket));
    free(sock);
  }
  *recvComm = rComm;

  /* reset lComm state */
  stage->state = ncclNetSocketCommStateStart;
  stage->iteration = 0;
  stage->sock = NULL;
  stage->comm = NULL;
  return ncclSuccess;
}

ncclResult_t ncclNetSocketGetRequest(struct ncclNetSocketComm* comm, int op, void* data, int size, struct ncclNetSocketRequest** req) {
  for (int i=0; i<MAX_REQUESTS; i++) {
    struct ncclNetSocketRequest* r = comm->requests+i;
    if (r->used == 0) {
      r->op = op;
      r->data = data;
      r->size = size;
      r->ctrlSock = &comm->ctrlSock;
      r->used = 1;
      r->comm = comm;
      r->nSubs = 0;
      *req = r;
      return ncclSuccess;
    }
  }
  WARN("NET/Socket : unable to allocate requests");
  return ncclInternalError;
}

ncclResult_t ncclNetSocketGetTask(struct ncclNetSocketComm* comm, int op, void* data, int size, struct ncclNetSocketTask** req) {
  int tid = comm->nextSock % comm->nThreads;
  struct ncclNetSocketThreadResources* res = comm->threadResources+tid;
  struct ncclNetSocketTaskQueue* queue = &res->threadTaskQueue;
  // create helper threads and prepare per-thread task queue
  if (queue->tasks == NULL) {
    // each request can be divided up to nSocks tasks, and
    // these tasks are distributed to nThreads threads,
    // we need to make sure each thread queue has enough slots for MAX_REQUESTS
    queue->len = MAX_REQUESTS * DIVUP(comm->nSocks, comm->nThreads);
    NCCLCHECK(ncclCalloc(&queue->tasks, queue->len));
    queue->next = 0;
    res->comm = comm;
    pthread_mutex_init(&res->threadLock, NULL);
    pthread_cond_init(&res->threadCond, NULL);
    PTHREADCHECK(pthread_create(comm->helperThread+tid, NULL, persistentSocketThread, res), "pthread_create");
    ncclSetThreadName(comm->helperThread[tid], "NCCL Sock%c%1u%2u%2u", op == NCCL_SOCKET_SEND ? 'S' : 'R', comm->dev, tid, comm->cudaDev);
  }
  struct ncclNetSocketTask* r = queue->tasks+queue->next;
  if (r->used == 0) {
    r->op = op;
    r->data = data;
    r->size = size;
    r->sock = comm->socks + comm->nextSock;
    r->offset = 0;
    r->result = ncclSuccess;
    comm->nextSock = (comm->nextSock + 1) % comm->nSocks;
    r->used = 1;
    *req = r;
    pthread_mutex_lock(&res->threadLock);
    queue->next = (queue->next+1)%queue->len;
    pthread_cond_signal(&res->threadCond);
    pthread_mutex_unlock(&res->threadLock);
    return ncclSuccess;
  }
  WARN("NET/Socket : unable to allocate subtasks");
  return ncclInternalError;
}

ncclResult_t ncclNetSocketTest(void* request, int* done, int* size) {
  *done = 0;
  struct ncclNetSocketRequest *r = (struct ncclNetSocketRequest*)request;
  if (r == NULL) {
    WARN("NET/Socket : test called with NULL request");
    return ncclInternalError;
  }
  if (r->used == 1) { /* try to send/recv size */
    int data = r->size;
    int offset = 0;
    NCCLCHECK(ncclSocketProgress(r->op, r->ctrlSock, &data, sizeof(int), &offset));

    if (offset == 0) return ncclSuccess; /* Not ready -- retry later */

    // Not sure we could ever receive less than 4 bytes, but just in case ...
    if (offset < sizeof(int)) NCCLCHECK(ncclSocketWait(r->op, r->ctrlSock, &data, sizeof(int), &offset));

    // Check size is less or equal to the size provided by the user
    if (r->op == NCCL_SOCKET_RECV && data > r->size) {
      char line[SOCKET_NAME_MAXLEN+1];
      union ncclSocketAddress addr;
      NCCLCHECK(ncclSocketGetAddr(r->ctrlSock, &addr));
      WARN("NET/Socket : peer %s message truncated : receiving %d bytes instead of %d. If you believe your socket network is in healthy state, \
          there may be a mismatch in collective sizes or environment settings (e.g. NCCL_PROTO, NCCL_ALGO) between ranks",
          ncclSocketToString(&addr, line), data, r->size);
      return ncclInvalidUsage;
    }
    r->size = data;
    r->offset = 0;
    r->used = 2; // done exchanging size
    // divide into subtasks
    int chunkOffset = 0, i = 0;
    if (r->comm->nSocks > 0) {
      // each request can be divided up to nSocks tasks
      int taskSize = std::max(MIN_CHUNKSIZE, DIVUP(r->size, r->comm->nSocks));
      while (chunkOffset < r->size) {
        int chunkSize = std::min(taskSize, r->size-chunkOffset);
        NCCLCHECK(ncclNetSocketGetTask(r->comm, r->op, (char*)(r->data)+chunkOffset, chunkSize, r->tasks+i++));
        chunkOffset += chunkSize;
      }
    }
    r->nSubs = i;
  }
  if (r->used == 2) { // already exchanged size
    if (r->nSubs > 0) {
      int nCompleted = 0;
      for (int i=0; i<r->nSubs; i++) {
        struct ncclNetSocketTask* sub = r->tasks[i];
        if (sub->result != ncclSuccess) return sub->result;
        if (sub->offset == sub->size) nCompleted++;
      }
      if (nCompleted == r->nSubs) {
        if (size) *size = r->size;
        *done = 1;
        r->used = 0;
        for (int i=0; i<r->nSubs; i++) {
          struct ncclNetSocketTask* sub = r->tasks[i];
          sub->used = 0;
        }
      }
    } else { // progress request using main thread
      if (r->offset < r->size) {
        NCCLCHECK(ncclSocketProgress(r->op, r->ctrlSock, r->data, r->size, &r->offset));
      }
      if (r->offset == r->size) {
        if (size) *size = r->size;
        *done = 1;
        r->used = 0;
      }
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclNetSocketRegMr(void* comm, void* data, size_t size, int type, void** mhandle) {
  return (type != NCCL_PTR_HOST) ? ncclInternalError : ncclSuccess;
}
ncclResult_t ncclNetSocketDeregMr(void* comm, void* mhandle) { return ncclSuccess; }

ncclResult_t ncclNetSocketIsend(void* sendComm, void* data, size_t size, int tag, void* mhandle, void** request) {
  struct ncclNetSocketComm* comm = (struct ncclNetSocketComm*)sendComm;
  NCCLCHECK(ncclNetSocketGetRequest(comm, NCCL_SOCKET_SEND, data, (int) size, (struct ncclNetSocketRequest**)request));
  return ncclSuccess;
}

ncclResult_t ncclNetSocketIrecv(void* recvComm, int n, void** data, size_t* sizes, int* tags, void** mhandles, void** request) {
  struct ncclNetSocketComm* comm = (struct ncclNetSocketComm*)recvComm;
  if (n != 1) return ncclInternalError;
  NCCLCHECK(ncclNetSocketGetRequest(comm, NCCL_SOCKET_RECV, data[0], (int)sizes[0], (struct ncclNetSocketRequest**)request));
  return ncclSuccess;
}

ncclResult_t ncclNetSocketIflush(void* recvComm, int n, void** data, int* sizes, void** mhandles, void** request) {
  // We don't support CUDA pointers, so we don't need a flush operation
  return ncclInternalError;
}

ncclResult_t ncclNetSocketCloseListen(void* opaqueComm) {
  struct ncclNetSocketListenComm* comm = (struct ncclNetSocketListenComm*)opaqueComm;
  if (comm) {
    int ready;
    NCCLCHECK(ncclSocketReady(&comm->sock, &ready));
    if (ready) NCCLCHECK(ncclSocketClose(&comm->sock));
    free(comm);
  }
  return ncclSuccess;
}

ncclResult_t ncclNetSocketClose(void* opaqueComm) {
  struct ncclNetSocketComm* comm = (struct ncclNetSocketComm*)opaqueComm;
  if (comm) {
    for (int i=0; i<comm->nThreads; i++) {
      struct ncclNetSocketThreadResources* res = comm->threadResources+i;
      if (comm->helperThread[i]) {
        pthread_mutex_lock(&res->threadLock);
        res->stop = 1;
        pthread_cond_signal(&res->threadCond);
        pthread_mutex_unlock(&res->threadLock);
        PTHREADCHECK(pthread_join(comm->helperThread[i], NULL), "pthread_join");
      }
      free(res->threadTaskQueue.tasks);
    }
    int ready;
    NCCLCHECK(ncclSocketReady(&comm->ctrlSock, &ready));
    if (ready) NCCLCHECK(ncclSocketClose(&comm->ctrlSock));
    for (int i=0; i<comm->nSocks; i++) {
      NCCLCHECK(ncclSocketReady(&comm->socks[i], &ready));
      if (ready) NCCLCHECK(ncclSocketClose(&comm->socks[i]));
    }
    free(comm);
  }
  return ncclSuccess;
}

ncclNet_t ncclNetSocket = {
  "Socket",
  ncclNetSocketInit,
  ncclNetSocketDevices,
  ncclNetSocketGetProperties,
  ncclNetSocketListen,
  ncclNetSocketConnect,
  ncclNetSocketAccept,
  ncclNetSocketRegMr,
  NULL, // No DMA-BUF support
  ncclNetSocketDeregMr,
  ncclNetSocketIsend,
  ncclNetSocketIrecv,
  ncclNetSocketIflush,
  ncclNetSocketTest,
  ncclNetSocketClose,
  ncclNetSocketClose,
  ncclNetSocketCloseListen,
  NULL /* getDeviceMr */,
  NULL /* irecvConsumed */,
  NULL /* mergeDevices */
};
