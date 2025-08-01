/*************************************************************************
 * Copyright (c) 2016-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#define NDEBUG // Comment out during development only!
#include <cassert>
#include <cstddef>
#include <mutex>
#include <poll.h>
#include <unistd.h>

#include "alloc.h"
#include "checks.h"
#include "comm.h"
#include "nccl.h"
#include "utils.h"
#include "ras_internal.h"

// Type of a notification from a local NCCL thread.
typedef enum {
  RAS_ADD_RANKS = 0,
  RAS_TERMINATE = 1
} rasNotificationType;

// Used for communication from local NCCL threads to the RAS thread.
struct rasNotification {
  rasNotificationType type;
  union {
    struct {
      struct rasRankInit* ranks;
      int nranks;
    } addRanks;
  };
};
static_assert(sizeof(struct rasNotification) <= PIPE_BUF, "The rasNotification structure is too large");

// These ensure that we get only one RAS port/thread per process.
static std::mutex rasInitMutex;
static bool rasInitialized = false;
static int rasInitRefCount = 0;

// The RAS network listening socket of this RAS thread (random port).
struct ncclSocket rasNetListeningSocket;

static pthread_t rasThread;

// Used for communication from regular NCCL threads to the RAS thread.
static std::mutex rasNotificationMutex;
static int rasNotificationPipe[2] = {-1, -1};

// Data for the main poll() in the RAS thread.
struct pollfd* rasPfds;
static int nRasPfds;

// We use it all over the place; no point in wasting the stack...
char rasLine[SOCKET_NAME_MAXLEN+1];

// An array holding the addresses of all NCCL communicators.  Modified by the NCCL threads (hence the mutex), read by
// the RAS thread.
std::mutex ncclCommsMutex;//用于保护对通信器数组的并发访问，因为多个NCCL线程可能同时修改这个数组
struct ncclComm** ncclComms = nullptr;//一个指向通信器指针的数组
int nNcclComms = 0;//记录了数组的当前大小
bool ncclCommsSorted = false; // Whether the array is currently sorted. We sort by the comms' commHash and rank.

static ncclResult_t rasLocalNotify(const struct rasNotification* msg);
static ncclResult_t rasLocalHandle();
static void rasLocalHandleTerminate();

static ncclResult_t rasMsgHandleConnInit(const struct rasMsg* msg, struct rasSocket* sock);
static ncclResult_t rasMsgHandleConnInitAck(const struct rasMsg* msg, struct rasSocket* sock);
static ncclResult_t rasNetSendNack(struct rasSocket* sock);

static void* rasThreadMain(void*);

NCCL_PARAM(RasTimeoutFactor, "RAS_TIMEOUT_FACTOR", 1);

//////////////////////////////////////////////////
// Functions invoked from regular NCCL threads. //
//////////////////////////////////////////////////

// Invoked by regular NCCL threads on every comm initialization.  This is the first function to call.
// The myRank structure should be passed with the addr element initialized to the IP address of the bootstrap
// network interface to use.  On a successful return, the address will be updated with the port number of the
// RAS network listening socket.  每个通信器(communicator)初始化时都会调用它。这个函数负责设置RAS线程和相关资源。
//创建了RAS线程、监听套接字（方便其他的进程连接）。一个RAS线程可能管理多个通信器。
ncclResult_t ncclRasCommInit(struct ncclComm* comm, struct rasRankInit* myRank) {
  ncclResult_t ret = ncclSuccess;
  if (!rasInitialized) {
    // 如果RAS尚未初始化，创建监听套接字、管道和RAS线程, 确保每个进程只有一个RAS线程和端口
    std::lock_guard<std::mutex> lock(rasInitMutex);
    if (!rasInitialized) {
      union ncclSocketAddress addr;

      memcpy(&addr, &myRank->addr, sizeof(addr));
      (addr.sa.sa_family == AF_INET ? addr.sin.sin_port : addr.sin6.sin6_port) = htons(0);// 设置端口为0，让系统自动分配端口
      NCCLCHECKGOTO(ncclSocketInit(&rasNetListeningSocket, &addr, NCCL_SOCKET_MAGIC, ncclSocketTypeRasNetwork,
                                   /*abortFlag*/nullptr, /*asyncFlag*/1), ret, fail);
      NCCLCHECKGOTO(ncclSocketListen(&rasNetListeningSocket), ret, fail);// 开始监听
      INFO(NCCL_RAS, "RAS network listening socket at %s",
           ncclSocketToString(&rasNetListeningSocket.addr, rasLine));

      (void)rasClientInitSocket();// 初始化客户端套接字，同时还bind和listen了

      SYSCHECKGOTO(pipe(rasNotificationPipe), "pipe", ret, fail);// 创建通知管道，用于NCCL线程与RAS线程通信。
// 创建RAS线程
      PTHREADCHECKGOTO(pthread_create(&rasThread, nullptr, &rasThreadMain, nullptr), "pthread_create", ret, fail);
      ncclSetThreadName(rasThread, "NCCL RAS");
      (void)pthread_detach(rasThread);//主线程调用 pthread_join 会阻塞，直到目标线程结束。这在某些场景（如异步任务）中不必要，甚至可能成为性能瓶颈。所以NCCL这里基本上都detach

      rasInitialized = true;// 标记RAS已初始化
    }
  }
  ncclAtomicRefCountIncrement(&rasInitRefCount);// 增加引用计数。一个进程可以创建多个通信器。多个通信器共享同一个RAS线程和端口，引用计数确保只有一个RAS实例

  {
    std::lock_guard<std::mutex> lock(ncclCommsMutex);

    int i;
    for (i = 0; i < nNcclComms; i++) {
      if (ncclComms[i] == nullptr)
        break;
    }
    if (i == nNcclComms) {
      NCCLCHECK(ncclRealloc(&ncclComms, nNcclComms, nNcclComms+RAS_INCREMENT*8)); // 如果数组已满，扩展数组
      nNcclComms += RAS_INCREMENT*8;
    }
    ncclComms[i] = comm;
    ncclCommsSorted = false;
  }

  if (myRank != nullptr)// 更新调用者的地址信息。因为最开始myRank->addr的端口号我们写的是0，所以后面需要更新为自动分配的端口号。其他节点上的进程需要使用这个地址和端口
    memcpy(&myRank->addr, &rasNetListeningSocket.addr, sizeof(myRank->addr)); //RAS网络监听套接字的地址信息（包括系统自动分配的端口号）回传给调用者。

exit:
  return ret;
fail:
  if (rasNotificationPipe[1] != 0)//- 关闭标准输入（0）会导致最严重的问题，因为许多程序依赖标准输入来接收用户输入;关闭标准输出（1）或标准错误（2）通常不会导致程序崩溃，最多只是丢失输出信息
    (void)close(rasNotificationPipe[1]);
  if (rasNotificationPipe[0] != 0)
    (void)close(rasNotificationPipe[0]);
  (void)close(rasClientListeningSocket);
  (void)ncclSocketClose(&rasNetListeningSocket);
  goto exit;
}

// Invoked by regular NCCL threads on every comm termination.
ncclResult_t ncclRasCommFini(const struct ncclComm* comm) {
  if (!rasInitialized)
    return ncclSuccess;
  {
    std::lock_guard<std::mutex> lock(ncclCommsMutex);
    for (int i = 0; i < nNcclComms; i++) {
      if (ncclComms[i] == comm) {
        ncclComms[i] = nullptr;
        ncclCommsSorted = false;
        break;
      }
    }
  }
  if (ncclAtomicRefCountDecrement(&rasInitRefCount) == 0) {
    struct rasNotification msg;
    msg.type = RAS_TERMINATE;
    NCCLCHECK(rasLocalNotify(&msg));
  }
  return ncclSuccess;
}

// Invoked by regular NCCL threads on every (non-split) comm initialization.  Provides info on all the ranks within
// the communicator.
ncclResult_t ncclRasAddRanks(struct rasRankInit* ranks, int nranks) {
  struct rasNotification msg;
  msg.type = RAS_ADD_RANKS;
  msg.addRanks.ranks = ranks;
  msg.addRanks.nranks = nranks;
  NCCLCHECK(rasLocalNotify(&msg));
  return ncclSuccess;
}

// Internal function running on regular NCCL threads -- asynchronously notifies the RAS thread.
static ncclResult_t rasLocalNotify(const struct rasNotification* msg) {
  if (!rasInitialized)
    return ncclSuccess;

  // Take an exclusive lock here to avoid multiplexing between multiple user threads (not sure if it's
  // strictly required, but it won't hurt)...
  std::lock_guard<std::mutex> lock(rasNotificationMutex);
  size_t done = 0;
  while (done < sizeof(*msg)) {
    ssize_t written;
    SYSCHECK(written = write(rasNotificationPipe[1], (char*)msg + done, sizeof(*msg) - done), "write");
    done += written;
  }
  return ncclSuccess;
}


/////////////////////////////////////////////////////////////////////////////////
// Functions related to the handling of local notifications from NCCL threads. //
/////////////////////////////////////////////////////////////////////////////////

// Handles asynchronous local notifications arriving from regular NCCL threads.
static ncclResult_t rasLocalHandle() {
  struct rasNotification msg;

  size_t done = 0;
  while (done < sizeof(msg)) {
    ssize_t nread;
    SYSCHECK(nread = read(rasNotificationPipe[0], (char*)&msg + done, sizeof(msg) - done), "read");
    if (nread == 0) // EOF
      return ncclSystemError;
    done += nread;
  }

  if (msg.type == RAS_ADD_RANKS) {
    NCCLCHECK(rasLocalHandleAddRanks(msg.addRanks.ranks, msg.addRanks.nranks));
  } else if (msg.type == RAS_TERMINATE) {
    rasLocalHandleTerminate();
  } else {
    WARN("RAS received unknown notification type %d", msg.type);
    return ncclInternalError;
  }

  return ncclSuccess;
}

// Handles local RAS_TERMINATE notification.
static void rasLocalHandleTerminate() {
  INFO(NCCL_RAS, "RAS handling local termination request");
  // For now we don't do anything.
}


////////////////////////////////////////////////
// Generic functions related to RAS messages. //
////////////////////////////////////////////////

// Allocates a RAS message of the desired length for sending.
// Behind the scenes allocates encapsulating rasMsgMeta structure, which includes local metadata stored in front
// of the message.
// Must use rasMsgFree to free.
ncclResult_t rasMsgAlloc(struct rasMsg** msg, size_t msgLen) {
  struct rasMsgMeta* meta = nullptr;
  NCCLCHECK(ncclCalloc((char**)&meta, offsetof(struct rasMsgMeta, msg) + msgLen));
  *msg = &meta->msg;
  // coverity[leaked_storage:FALSE] => rasMsgFree is used to free it
  return ncclSuccess;
}

// To be used only with messages allocated with rasMsgAlloc.  I.e., it should be used for sent messages, not
// for received ones.
void rasMsgFree(struct rasMsg* msg) {
  if (msg) {
    struct rasMsgMeta* meta = (struct rasMsgMeta*)((char*)msg - offsetof(struct rasMsgMeta, msg));
    free(meta);
  }
}

// Enqueues a message for sending down a RAS connection.
void rasConnEnqueueMsg(struct rasConnection* conn, struct rasMsg* msg, size_t msgLen, bool front) {
  // Get to the metadata of this message.
  struct rasMsgMeta* meta = (struct rasMsgMeta*)((char*)msg - offsetof(struct rasMsgMeta, msg));
  bool ready = false;

  meta->enqueueTime = clockNano();
  meta->offset = 0;
  meta->length = msgLen;

  if (front)
    ncclIntruQueueEnqueueFront(&conn->sendQ, meta);
  else
    ncclIntruQueueEnqueue(&conn->sendQ, meta);

  if (conn->sockIdx != -1) {
    struct rasSocket* sock = rasSockets+conn->sockIdx;
    if (sock->status == RAS_SOCK_READY || (sock->status == RAS_SOCK_HANDSHAKE && msg->type == RAS_MSG_CONNINIT)) {
      rasPfds[sock->pfd].events |= POLLOUT;
      ready = true;
    }
  }
  if (!ready) {
    // It's not a bug, unless it's for things like keep-alive messages...
    INFO(NCCL_RAS, "RAS enqueued message type %d on a non-ready connection with %s "
         "(experiencingDelays %d, startRetryTime %.2fs, socket status %d)",
         msg->type, ncclSocketToString(&conn->addr, rasLine),
         conn->experiencingDelays, (conn->startRetryTime ? (clockNano()-conn->startRetryTime)/1e9 : 0.0),
         (conn->sockIdx == -1 ? -1 : rasSockets[conn->sockIdx].status));
  }
}

// Attempts to send the queued RAS messages to another RAS thread.
ncclResult_t rasConnSendMsg(struct rasConnection* conn, int* closed, bool* allSent) {
  struct ncclSocket* sock = &rasSockets[conn->sockIdx].sock;
  struct rasMsgMeta* meta;
  *closed = 0;
  while ((meta = ncclIntruQueueHead(&conn->sendQ)) != nullptr) {
    if (rasSockets[conn->sockIdx].status == RAS_SOCK_HANDSHAKE && meta->msg.type != RAS_MSG_CONNINIT) {
      // We don't send anything beyond the handshake at this point.
      meta = nullptr;
      break;
    }
    if (meta->offset < sizeof(meta->length)) {
      // Send the length of the message.
      NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_SEND, sock, &meta->length, sizeof(meta->length), &meta->offset, closed));
      if (*closed)
        return ncclSuccess;
      if (meta->offset < sizeof(meta->length))
        break;
    }
    // Send the body of the message.
    NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_SEND, sock, ((char*)&meta->msg)-sizeof(meta->length),
                                 meta->length+sizeof(meta->length), &meta->offset, closed));
    if (*closed)
      return ncclSuccess;
    if (meta->offset < meta->length+sizeof(meta->length))
      break;
    ncclIntruQueueDequeue(&conn->sendQ);
    free(meta);
  }

  *allSent = !meta;

  return ncclSuccess;
}

// Attempts to receive a message through a RAS socket.
ncclResult_t rasMsgRecv(struct rasSocket* sock, struct rasMsg** msg, int* closed) {
  *closed = 0;
  if (sock->recvOffset < sizeof(sock->recvLength)) {
    // Receive the length of the message.
    NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_RECV, &sock->sock, &sock->recvLength, sizeof(sock->recvLength),
                                 &sock->recvOffset, closed));
    if (*closed || sock->recvOffset < sizeof(sock->recvLength))
      return ncclSuccess;
    NCCLCHECK(ncclCalloc((char**)&sock->recvMsg, sock->recvLength));
  }
  // Receive the body of the message.
  NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_RECV, &sock->sock, ((char*)sock->recvMsg)-sizeof(sock->recvLength),
                               sock->recvLength+sizeof(sock->recvLength), &sock->recvOffset, closed));
  if (*closed || sock->recvOffset < sock->recvLength+sizeof(sock->recvLength))
    return ncclSuccess;

  *msg = sock->recvMsg;
  sock->recvMsg = nullptr;
  sock->recvOffset = sock->recvLength = 0;

  return ncclSuccess;
}


//////////////////////////////////////////////////////////////////
// Functions related to the handling of specific message types. //
//////////////////////////////////////////////////////////////////

// Invoked from the main RAS thread to dispatch incoming messages to the appropriate handler.
ncclResult_t rasMsgHandle(struct rasMsg* msg, struct rasSocket* sock) {
  if (msg->type == RAS_MSG_CONNINIT) {
    NCCLCHECK(rasMsgHandleConnInit(msg, sock));
  } else if (msg->type == RAS_MSG_CONNINITACK) {
    NCCLCHECK(rasMsgHandleConnInitAck(msg, sock));
  } else if (msg->type == RAS_MSG_KEEPALIVE) {
    NCCLCHECK(rasMsgHandleKeepAlive(msg, sock));
  } else if (msg->type == RAS_MSG_PEERSUPDATE) {
    NCCLCHECK(rasMsgHandlePeersUpdate(msg, sock));
  } else if (msg->type == RAS_MSG_COLLREQ) {
    NCCLCHECK(rasMsgHandleCollReq(msg, sock));
  } else if (msg->type == RAS_MSG_COLLRESP) {
    NCCLCHECK(rasMsgHandleCollResp(msg, sock));
  } else {
    WARN("RAS received unknown message type (%d) from %s", msg->type, ncclSocketToString(&sock->sock.addr, rasLine));
    return ncclInternalError;
  }

  return ncclSuccess;
}

// Handles the first message sent over a RAS socket as part of the handshake.
static ncclResult_t rasMsgHandleConnInit(const struct rasMsg* msg, struct rasSocket* sock) {
  ncclResult_t ret = ncclSuccess;
  struct rasConnection* conn = nullptr;
  int connIdx, peerIdx;
  struct rasMsg* newMsg = nullptr;
  int newMsgLen;
  char line[SOCKET_NAME_MAXLEN+1];

  INFO(NCCL_RAS, "RAS handling connInit from %s (version %d, listeningAddr %s, peersHash 0x%lx, deadPeersHash 0x%lx)",
       ncclSocketToString(&sock->sock.addr, rasLine), msg->connInit.ncclVersion,
       ncclSocketToString(&msg->connInit.listeningAddr, line), msg->connInit.peersHash, msg->connInit.deadPeersHash);

  if (msg->connInit.ncclVersion != NCCL_VERSION_CODE) {
    // Close any such sockets immediately!  This is basically unrecoverable...
    WARN("NCCL version mismatch with remote peer %s (local: %d, remote %d)",
         ncclSocketToString(&sock->sock.addr, rasLine), NCCL_VERSION_CODE, msg->connInit.ncclVersion);
    rasNetSendNack(sock);
    rasSocketTerminate(sock, /*finalize*/true);
    ret = ncclInvalidUsage;
    goto exit;
  }

  if (rasPeerIsDead(&msg->connInit.listeningAddr)) {
    // A peer long declared dead is suddenly alive again?!
    INFO(NCCL_RAS, "RAS connection from peer %s that is considered dead!",
         ncclSocketToString(&msg->connInit.listeningAddr, rasLine));
    rasNetSendNack(sock);
    rasSocketTerminate(sock, /*finalize*/true);
    goto exit;
  }

  // Check for any existing connection with that RAS thread (could happen due to a network issue, or possibly a race).
  connIdx = rasConnFind(&msg->connInit.listeningAddr);
  if (connIdx != -1) {
    conn = rasConns+connIdx;

    INFO(NCCL_RAS,
         "RAS found a matching existing connection (sendQ %sempty, experiencingDelays %d, startRetryTime %.2fs)",
         (ncclIntruQueueEmpty(&conn->sendQ) ? "" : "not "),
         conn->experiencingDelays, (conn->startRetryTime ? (clockNano()-conn->startRetryTime)/1e9 : 0.0));

    if (conn->sockIdx != -1) {
      struct rasSocket* connSock = rasSockets+conn->sockIdx;
      INFO(NCCL_RAS, "RAS found an alternative existing socket (status %d, createTime %.2fs)",
           connSock->status, (clockNano()-connSock->createTime)/1e9);
      // In general we prefer to keep the newer connection, but "newer" can be a relative term: we may have
      // a race where both sides attempt to establish a connection at roughly the same time, so the other side's
      // incoming connection ends up looking newer than the locally-initiated one -- for *both* of them.
      // If each side closed the "old" one, both would end up being closed.
      // As we normally try to initiate connections from the side with a lower address (precisely to avoid such
      // situations), we'll follow the same logic here: the "lower" side will reject the new connection (as it
      // came from the "wrong" side), whereas the "higher" side will keep the new one (as it came from the correct
      // side) and terminate the old one (that it presumably just opened).
      if (ncclSocketsCompare(&rasNetListeningSocket.addr, &conn->addr) < 0) {
        INFO(NCCL_RAS, "RAS terminating the new socket");
        rasSocketTerminate(sock, /*finalize*/true);
        goto exit;
      } else {
        INFO(NCCL_RAS, "RAS keeping the new socket and terminating the existing one");
        rasSocketTerminate(connSock);
      }
    }
  }
  if (!conn) {
    NCCLCHECK(getNewConnEntry(&conn));
    memcpy(&conn->addr, &msg->connInit.listeningAddr, sizeof(conn->addr));
    connIdx = conn - rasConns;
  }

  sock->status = RAS_SOCK_READY;
  // rasConnResume will reset any experiencingDelays, startRetryTime, etc.

  conn->sockIdx = sock-rasSockets;
  sock->connIdx = connIdx;
  memcpy(&sock->sock.addr, &msg->connInit.listeningAddr, sizeof(sock->sock.addr));

  // Make sure that the connection is part of the right links forming the RAS network.  At this point we only
  // update the expected (non-external) connections; external ones will be added during keep-alive handling.
  peerIdx = rasPeerFind(&conn->addr);
  // Note: it's possible for peerIdx to be -1 at this point if, due to races, the connInit arrives before
  // the peers update.
  if (peerIdx != -1) {
    (void)rasLinkUpdateConn(&rasNextLink, connIdx, peerIdx);
    (void)rasLinkUpdateConn(&rasPrevLink, connIdx, peerIdx);
  }

  // Send a confirmation to the server that requested the connection (so that the resilience code can mark
  // the connection as live).
  newMsgLen = rasMsgLength(RAS_MSG_CONNINITACK);
  NCCLCHECK(rasMsgAlloc(&newMsg, newMsgLen));
  newMsg->type = RAS_MSG_CONNINITACK;
  newMsg->connInitAck.nack = 0;
  rasConnEnqueueMsg(conn, newMsg, newMsgLen, /*front*/true);

  conn->lastRecvPeersHash = msg->connInit.peersHash;
  conn->lastRecvDeadPeersHash = msg->connInit.deadPeersHash;

  if (msg->connInit.peersHash != rasPeersHash || msg->connInit.deadPeersHash != rasDeadPeersHash) {
    // Send my rasPeers and request the same in return.
    INFO(NCCL_RAS, "RAS connInit hash mismatch (my peersHash 0x%lx, deadPeersHash 0x%lx); sending my (dead) peers",
         rasPeersHash, rasDeadPeersHash);
    NCCLCHECK(rasConnSendPeersUpdate(conn, rasPeers, nRasPeers));
  }
exit:
  return ret;
}

// Handles the second message sent over a RAS socket as part of the handshake.
static ncclResult_t rasMsgHandleConnInitAck(const struct rasMsg* msg, struct rasSocket* sock) {
  INFO(NCCL_RAS, "RAS handling connInitAck from %s (nack %d)",
       ncclSocketToString(&sock->sock.addr, rasLine), msg->connInitAck.nack);

  if (msg->connInitAck.nack) {
    // The remote peer doesn't want to talk to us.  The easiest way to prevent it is by declaring it dead.
    // We make a copy of the address because rasConnDisconnect will terminate the rasSocket.
    union ncclSocketAddress addr;
    memcpy(&addr, &sock->sock.addr, sizeof(addr));
    rasConnDisconnect(&addr);
    (void)rasPeerDeclareDead(&addr);

    return ncclSuccess;
  }

  sock->status = RAS_SOCK_READY;
  // rasConnResume will reset any experiencingDelays, startRetryTime, etc.

  return ncclSuccess;
}

// Handles the deadPeer broadcast.
void rasMsgHandleBCDeadPeer(const struct rasCollRequest* req, bool* pDone) {
  INFO(NCCL_RAS, "RAS handling deadPeer (addr %s)", ncclSocketToString(&req->deadPeer.addr, rasLine));

  if (!rasPeerIsDead(&req->deadPeer.addr)) {
    rasConnDisconnect(&req->deadPeer.addr);
    (void)rasPeerDeclareDead(&req->deadPeer.addr);
    *pDone = false;
  } else {
    INFO(NCCL_RAS, "RAS already knew it was dead");
    // No point in re-broadcasting what's already known.
    *pDone = true;
  }
}

// Attempts to immediately send a fatal NACK connInitAck response to a socket.  A bit of a hack (as it doesn't
// follow our usual message queuing and polling convention) but, since this can be invoked only for newly opened
// connections, and the message is tiny, it should be OK.  We can't use the regular path because the socket is
// about to be terminated.
static ncclResult_t rasNetSendNack(struct rasSocket* sock) {
  struct rasMsg msg;
  int length = rasMsgLength(RAS_MSG_CONNINITACK);
  int closed = 0;
  int offset;

  INFO(NCCL_RAS, "RAS sending NACK to %s", ncclSocketToString(&sock->sock.addr, rasLine));

  msg.type = RAS_MSG_CONNINITACK;
  msg.connInitAck.nack = 1;
  offset = 0;
  NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_SEND, &sock->sock, &length, sizeof(length), &offset, &closed));
  if (closed || offset < sizeof(length))
    return ncclSuccess;
  offset = 0;
  NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_SEND, &sock->sock, &msg, length, &offset, &closed));
  // We are closing this socket anyway -- it doesn't matter to us if we succeeded or not.

  return ncclSuccess;
}


/////////////////////////////////////////////////////////////////
// Functions related to the main event loop of the RAS thread. //
/////////////////////////////////////////////////////////////////

// Main function of the RAS thread.
static void* rasThreadMain(void*) {
  ncclResult_t ret = ncclSuccess; // Unused.
  int pfd;
  int rasNetListeningSocketFd;

  INFO(NCCL_RAS, "RAS thread started");

  // Initialize the global pollfd with the file descriptors we already have (the pipe and the listening socket).
  NCCLCHECKGOTO(rasGetNewPollEntry(&pfd), ret, fail);
  rasPfds[pfd].fd = rasNotificationPipe[0];
  rasPfds[pfd].events = POLLIN;

  NCCLCHECKGOTO(rasGetNewPollEntry(&pfd), ret, fail);
  NCCLCHECKGOTO(ncclSocketGetFd(&rasNetListeningSocket, &rasNetListeningSocketFd), ret, fail);
  rasPfds[pfd].fd = rasNetListeningSocketFd;
  rasPfds[pfd].events = POLLIN;

  NCCLCHECKGOTO(rasGetNewPollEntry(&pfd), ret, fail);
  rasPfds[pfd].fd = rasClientListeningSocket;
  rasPfds[pfd].events = POLLIN;

  // Main event loop of the RAS thread.
  for (int64_t nextWakeup=0;;) {
    int timeout, nEvents;
    int64_t now = clockNano();
    if (nextWakeup > 0) {
      // The "1" below helps avoid round-downs and especially zeroes.
      if (nextWakeup > now)
        timeout = (nextWakeup - now) / (CLOCK_UNITS_PER_SEC / 1000) + 1;
      else
        timeout = 1;
    } else {
      timeout = 1000; // 1 second.
    }

    nEvents = poll(rasPfds, nRasPfds, timeout);

    nextWakeup = clockNano()+CLOCK_UNITS_PER_SEC;
    if (nEvents == -1 && errno != EINTR)
      INFO(NCCL_RAS, "RAS continuing in spite of an unexpected error from poll: %s", strerror(errno));

    // Handle any poll-related events.
    for (int pollIdx = 0; pollIdx < nRasPfds && nEvents > 0; pollIdx++) {
      if (rasPfds[pollIdx].revents) {
        nEvents--;
        if (rasPfds[pollIdx].fd == rasNotificationPipe[0]) {
          (void)rasLocalHandle();
        } else if (rasPfds[pollIdx].fd == rasNetListeningSocketFd) {
          (void)rasNetAcceptNewSocket();
        } else if (rasPfds[pollIdx].fd == rasClientListeningSocket) {
          (void)rasClientAcceptNewSocket();
        } else {
          // Check if it's one of the RAS sockets.
          int sockIdx;
          for (sockIdx = 0; sockIdx < nRasSockets; sockIdx++) {
            struct rasSocket* sock = rasSockets+sockIdx;
            if (sock->status != RAS_SOCK_CLOSED && rasPfds[pollIdx].fd == sock->sock.fd) {
              rasSockEventLoop(sockIdx, pollIdx);
              break;
            }
          } // for (sockIdx)

          if (sockIdx == nRasSockets) {
            // Try a client socket instead.
            for (int clientIdx = 0; clientIdx < nRasClients; clientIdx++) {
              struct rasClient* client = rasClients+clientIdx;
              if (client->status != RAS_CLIENT_CLOSED && rasPfds[pollIdx].fd == client->sock) {
                rasClientEventLoop(clientIdx, pollIdx);
                break;
              }
            } // for (clientIdx)
          } // if (sockIdx == nRasSockets)
        } // dynamic fds
      } // if (revents)
    } // for (pollIdx)

    now = clockNano();

    rasSocksHandleTimeouts(now, &nextWakeup);

    rasConnsHandleTimeouts(now, &nextWakeup);

    rasNetHandleTimeouts(now, &nextWakeup);

    rasCollsHandleTimeouts(now, &nextWakeup);
  } // for (;;)

fail:
  WARN("fatal error - RAS thread terminating");
  std::lock_guard<std::mutex> lock(rasInitMutex);
  (void)close(rasNotificationPipe[1]);
  (void)close(rasNotificationPipe[0]);
  (void)close(rasClientListeningSocket);
  (void)ncclSocketClose(&rasNetListeningSocket);
  rasInitialized = false;
  return nullptr;
}

// Returns the index of the first available entry in the rasPfds array, enlarging the array if necessary.
ncclResult_t rasGetNewPollEntry(int* index) {
  int i;
  for (i = 0; i < nRasPfds; i++)
    if (rasPfds[i].fd == -1)
      break;
  if (i == nRasPfds) {
    NCCLCHECK(ncclRealloc(&rasPfds, nRasPfds, nRasPfds+RAS_INCREMENT));
    nRasPfds += RAS_INCREMENT;
    for (int j = i; j < nRasPfds; j++)
      rasPfds[j].fd = -1;
  }

  memset(rasPfds+i, '\0', sizeof(*rasPfds));
  rasPfds[i].fd = -1;

  *index = i;
  return ncclSuccess;
}
