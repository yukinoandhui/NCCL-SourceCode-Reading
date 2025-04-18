/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_SOCKET_H_
#define NCCL_SOCKET_H_

#include "nccl.h"
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <netdb.h>
#include <fcntl.h>
#include <poll.h>

#define MAX_IFS 16
#define MAX_IF_NAME_SIZE 16
#define SOCKET_NAME_MAXLEN (NI_MAXHOST+NI_MAXSERV)
#define NCCL_SOCKET_MAGIC 0x564ab9f2fc4b9d6cULL

/* Common socket address storage structure for IPv4/IPv6 */
union ncclSocketAddress {
  struct sockaddr sa;
  struct sockaddr_in sin;
  struct sockaddr_in6 sin6;
};

enum ncclSocketState {
  ncclSocketStateNone = 0,
  ncclSocketStateInitialized = 1,
  ncclSocketStateAccepting = 2,
  ncclSocketStateAccepted = 3,
  ncclSocketStateConnecting = 4,
  ncclSocketStateConnectPolling = 5,
  ncclSocketStateConnected = 6,
  ncclSocketStateReady = 7,
  ncclSocketStateTerminating = 8,
  ncclSocketStateClosed = 9,
  ncclSocketStateError = 10,
  ncclSocketStateNum = 11
};

enum ncclSocketType {
  ncclSocketTypeUnknown = 0,// 未知类型，默认值
  ncclSocketTypeBootstrap = 1,// 用于引导过程的套接字
  ncclSocketTypeProxy = 2,// 用于代理服务的套接字
  ncclSocketTypeNetSocket = 3,// 网络套接字
  ncclSocketTypeNetIb = 4,// InfiniBand网络套接字
  ncclSocketTypeRasNetwork = 5// 可靠性、可用性和可服务性(RAS)网络套接字
};

struct ncclSocket {
  int fd;// 文件描述符，用于标识socket连接
  int acceptFd;// 用于accept操作的临时文件描述符，就是accept的参数，accept返回的则是fd
  int errorRetries;// 错误重试次数计数器
  union ncclSocketAddress addr;// 套接字地址信息（支持IPv4/IPv6）
  volatile uint32_t* abortFlag;// 中止标志指针，用于异步中断操作
  int asyncFlag;// 异步操作标志
  enum ncclSocketState state;// 套接字当前状态
  int salen;// 套接字地址长度
  uint64_t magic;// 魔数，用于验证连接的有效性
  enum ncclSocketType type;// 套接字类型（Bootstrap/Proxy等）
  int customRetry;// 自定义重试策略标志
  int finalizeCounter; // Used to keep track of initial handshake for async sockets.// 异步套接字初始握手计数器
  char finalizeBuffer[sizeof(uint64_t)]; // Used to keep track of initial handshake for async sockets. // 初始握手数据缓冲区
};

const char *ncclSocketToString(const union ncclSocketAddress *addr, char *buf, const int numericHostForm = 1);
ncclResult_t ncclSocketGetAddrFromString(union ncclSocketAddress* ua, const char* ip_port_pair);
int ncclFindInterfaceMatchSubnet(char* ifNames, union ncclSocketAddress* localAddrs, union ncclSocketAddress* remoteAddr, int ifNameMaxSize, int maxIfs);
int ncclFindInterfaces(char* ifNames, union ncclSocketAddress *ifAddrs, int ifNameMaxSize, int maxIfs);

// Initialize a socket
ncclResult_t ncclSocketInit(struct ncclSocket* sock, const union ncclSocketAddress* addr = NULL, uint64_t magic = NCCL_SOCKET_MAGIC, enum ncclSocketType type = ncclSocketTypeUnknown, volatile uint32_t* abortFlag = NULL, int asyncFlag = 0, int customRetry = 0);
// Create a listening socket. sock->addr can be pre-filled with IP & port info. sock->fd is set after a successful call
ncclResult_t ncclSocketListen(struct ncclSocket* sock);
ncclResult_t ncclSocketGetAddr(struct ncclSocket* sock, union ncclSocketAddress* addr);
// Connect to sock->addr. sock->fd is set after a successful call.
ncclResult_t ncclSocketConnect(struct ncclSocket* sock);
// Return socket connection state.
ncclResult_t ncclSocketReady(struct ncclSocket* sock, int *running);
// Accept an incoming connection from listenSock->fd and keep the file descriptor in sock->fd, with the remote side IP/port in sock->addr.
ncclResult_t ncclSocketAccept(struct ncclSocket* sock, struct ncclSocket* ulistenSock);
ncclResult_t ncclSocketGetFd(struct ncclSocket* sock, int* fd);
ncclResult_t ncclSocketSetFd(int fd, struct ncclSocket* sock);

#define NCCL_SOCKET_SEND 0
#define NCCL_SOCKET_RECV 1

ncclResult_t ncclSocketProgress(int op, struct ncclSocket* sock, void* ptr, int size, int* offset, int* closed = NULL);
ncclResult_t ncclSocketWait(int op, struct ncclSocket* sock, void* ptr, int size, int* offset);
ncclResult_t ncclSocketSend(struct ncclSocket* sock, void* ptr, int size);
ncclResult_t ncclSocketRecv(struct ncclSocket* sock, void* ptr, int size);
ncclResult_t ncclSocketSendRecv(struct ncclSocket* sendSock, void* sendPtr, int sendSize, struct ncclSocket* recvSock, void* recvPtr, int recvSize);
ncclResult_t ncclSocketTryRecv(struct ncclSocket* sock, void* ptr, int size, int* closed, bool blocking);
ncclResult_t ncclSocketShutdown(struct ncclSocket* sock, int how);
ncclResult_t ncclSocketClose(struct ncclSocket* sock);
#endif
