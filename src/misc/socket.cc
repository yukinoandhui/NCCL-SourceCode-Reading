/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "socket.h"
#include "utils.h"
#include <stdlib.h>

#include <unistd.h>
#include <ifaddrs.h>
#include <net/if.h>
#include "param.h"
#include <time.h>

NCCL_PARAM(RetryCnt, "SOCKET_RETRY_CNT", 34);
NCCL_PARAM(RetryTimeOut, "SOCKET_RETRY_SLEEP_MSEC", 100);
static void msleep(unsigned int time_msec) {
  const long c_1e6 = 1e6;
  struct timespec tv = (struct timespec){
      .tv_sec = time_msec / 1000,
      .tv_nsec = (time_msec % 1000) * c_1e6,
  };
  nanosleep(&tv, NULL);
}
//尝试发送/接收数据。出错时或者在recv模式没有收到数据，会填充closed为1，表明远端关闭了连接
static ncclResult_t socketProgressOpt(int op, struct ncclSocket* sock, void* ptr, int size, int* offset, int block, int* closed) {
  int bytes = 0;
  *closed = 0;
  char* data = (char*)ptr;
  char line[SOCKET_NAME_MAXLEN+1];
  do {
    if (op == NCCL_SOCKET_RECV) bytes = recv(sock->fd, data+(*offset), size-(*offset), block ? 0 : MSG_DONTWAIT);//使用 MSG_DONTWAIT 实现非阻塞操作
    if (op == NCCL_SOCKET_SEND) bytes = send(sock->fd, data+(*offset), size-(*offset), block ? MSG_NOSIGNAL : MSG_DONTWAIT | MSG_NOSIGNAL);
    if (op == NCCL_SOCKET_RECV && bytes == 0) {
      *closed = 1;
      return ncclSuccess;
    }
    if (bytes == -1) {
      if ((op == NCCL_SOCKET_SEND && errno == EPIPE) || (op == NCCL_SOCKET_RECV && errno == ECONNRESET)) {
        *closed = 1;
        return ncclSuccess;
      }
      if (errno != EINTR && errno != EWOULDBLOCK && errno != EAGAIN) {
        WARN("socketProgressOpt: Call to %s %s failed : %s", (op == NCCL_SOCKET_RECV ? "recv from" : "send to"),
             ncclSocketToString(&sock->addr, line), strerror(errno));
        return ncclRemoteError;
      } else {
        bytes = 0;
      }
    }
    (*offset) += bytes;
    if (sock->abortFlag && __atomic_load_n(sock->abortFlag, __ATOMIC_ACQUIRE)) {
      INFO(NCCL_NET, "socketProgressOpt: abort called");
      return ncclInternalError;
    }
  } while (sock->asyncFlag == 0 && bytes > 0 && (*offset) < size);//如果不指定异步，那这里会一直阻塞（同步模式）。
  return ncclSuccess;
}
//根据指定op执行相应动作（从offset开始发送、接收），如果peer端关闭连接，则pclosed=1；
static ncclResult_t socketProgress(int op, struct ncclSocket* sock, void* ptr, int size, int* offset, int* pclosed = NULL) {
  int closed;
  NCCLCHECK(socketProgressOpt(op, sock, ptr, size, offset, 0 /*block*/, &closed));//固定使用非阻塞模式（block=0）
  if (closed) {
    if (pclosed) {
      *pclosed = closed;
      return ncclSuccess;
    } else {
      char line[SOCKET_NAME_MAXLEN+1];
      WARN("socketProgress: Connection closed by remote peer %s", ncclSocketToString(&sock->addr, line, 0));
      return ncclRemoteError;
    }
  }
  return ncclSuccess;
}
//发送size大小的数据
static ncclResult_t socketWait(int op, struct ncclSocket* sock, void* ptr, int size, int* offset) {
  while (*offset < size)
    NCCLCHECK(socketProgress(op, sock, ptr, size, offset));
  return ncclSuccess;
}

/* Format a string representation of a (union ncclSocketAddress *) socket address using getnameinfo()
 *
 * Output: "IPv4/IPv6 address<port>"
 */
const char *ncclSocketToString(const union ncclSocketAddress *addr, char *buf, const int numericHostForm /*= 1*/) {
  if (buf == NULL || addr == NULL) return NULL;
  const struct sockaddr *saddr = &addr->sa;
  if (saddr->sa_family != AF_INET && saddr->sa_family != AF_INET6) { buf[0]='\0'; return buf; }
  char host[NI_MAXHOST], service[NI_MAXSERV];
  /* NI_NUMERICHOST: If set, then the numeric form of the hostname is returned.
   * (When not set, this will still happen in case the node's name cannot be determined.)
   */
  int flag = NI_NUMERICSERV | (numericHostForm ? NI_NUMERICHOST : 0);
  (void) getnameinfo(saddr, sizeof(union ncclSocketAddress), host, NI_MAXHOST, service, NI_MAXSERV, flag);
  sprintf(buf, "%s<%s>", host, service);
  return buf;
}
//获取主机字节序的端口号
static uint16_t socketToPort(union ncclSocketAddress *addr) {
  struct sockaddr *saddr = &addr->sa;
  return ntohs(saddr->sa_family == AF_INET ? addr->sin.sin_port : addr->sin6.sin6_port);//将 16 位（2字节）数据从网络字节序（大端序）转换为主机字节序​​
}

/* Allow the user to force the IPv4/IPv6 interface selection */
static int envSocketFamily(void) {
  int family = -1; // Family selection is not forced, will use first one found
  const char* env = ncclGetEnv("NCCL_SOCKET_FAMILY");
  if (env == NULL)
    return family;

  INFO(NCCL_ENV, "NCCL_SOCKET_FAMILY set by environment to %s", env);

  if (strcmp(env, "AF_INET") == 0)
    family = AF_INET;  // IPv4
  else if (strcmp(env, "AF_INET6") == 0)
    family = AF_INET6; // IPv6
  return family;
}

static int findInterfaces(const char* prefixList, char* names, union ncclSocketAddress *addrs, int sock_family, int maxIfNameSize, int maxIfs) {
#ifdef ENABLE_TRACE
  char line[SOCKET_NAME_MAXLEN+1];
#endif
  struct netIf userIfs[MAX_IFS];
  bool searchNot = prefixList && prefixList[0] == '^';
  if (searchNot) prefixList++;
  bool searchExact = prefixList && prefixList[0] == '=';
  if (searchExact) prefixList++;
  int nUserIfs = parseStringList(prefixList, userIfs, MAX_IFS);

  int found = 0;
  struct ifaddrs *interfaces, *interface;
  getifaddrs(&interfaces);
  for (interface = interfaces; interface && found < maxIfs; interface = interface->ifa_next) {
    if (interface->ifa_addr == NULL) continue;

    /* We only support IPv4 & IPv6 */
    int family = interface->ifa_addr->sa_family;
    if (family != AF_INET && family != AF_INET6)
      continue;

    TRACE(NCCL_INIT|NCCL_NET,"Found interface %s:%s", interface->ifa_name, ncclSocketToString((union ncclSocketAddress *) interface->ifa_addr, line));

    /* Allow the caller to force the socket family type */
    if (sock_family != -1 && family != sock_family)
      continue;

    /* We also need to skip IPv6 loopback interfaces */
    if (family == AF_INET6) {
      struct sockaddr_in6* sa = (struct sockaddr_in6*)(interface->ifa_addr);
      if (IN6_IS_ADDR_LOOPBACK(&sa->sin6_addr)) continue;
    }

    // check against user specified interfaces
    if (!(matchIfList(interface->ifa_name, -1, userIfs, nUserIfs, searchExact) ^ searchNot)) {
      continue;
    }

    // Check that this interface has not already been saved
    // getifaddrs() normal order appears to be; IPv4, IPv6 Global, IPv6 Link
    bool duplicate = false;
    for (int i = 0; i < found; i++) {
      if (strcmp(interface->ifa_name, names+i*maxIfNameSize) == 0) { duplicate = true; break; }
    }

    if (!duplicate) {
      // Store the interface name
      strncpy(names+found*maxIfNameSize, interface->ifa_name, maxIfNameSize);
      // Store the IP address
      int salen = (family == AF_INET) ? sizeof(struct sockaddr_in) : sizeof(struct sockaddr_in6);
      memcpy(addrs+found, interface->ifa_addr, salen);
      found++;
    }
  }

  freeifaddrs(interfaces);
  return found;
}

static bool matchSubnet(struct ifaddrs local_if, union ncclSocketAddress* remote) {
  /* Check family first */
  int family = local_if.ifa_addr->sa_family;
  if (family != remote->sa.sa_family) {
    return false;
  }

  if (family == AF_INET) {
    struct sockaddr_in* local_addr = (struct sockaddr_in*)(local_if.ifa_addr);
    struct sockaddr_in* mask = (struct sockaddr_in*)(local_if.ifa_netmask);
    struct sockaddr_in& remote_addr = remote->sin;
    struct in_addr local_subnet, remote_subnet;
    local_subnet.s_addr = local_addr->sin_addr.s_addr & mask->sin_addr.s_addr;
    remote_subnet.s_addr = remote_addr.sin_addr.s_addr & mask->sin_addr.s_addr;
    return (local_subnet.s_addr ^ remote_subnet.s_addr) ? false : true;
  } else if (family == AF_INET6) {
    struct sockaddr_in6* local_addr = (struct sockaddr_in6*)(local_if.ifa_addr);
    struct sockaddr_in6* mask = (struct sockaddr_in6*)(local_if.ifa_netmask);
    struct sockaddr_in6& remote_addr = remote->sin6;
    struct in6_addr& local_in6 = local_addr->sin6_addr;
    struct in6_addr& mask_in6 = mask->sin6_addr;
    struct in6_addr& remote_in6 = remote_addr.sin6_addr;
    bool same = true;
    int len = 16;  //IPv6 address is 16 unsigned char
    for (int c = 0; c < len; c++) {  //Network byte order is big-endian
      char c1 = local_in6.s6_addr[c] & mask_in6.s6_addr[c];
      char c2 = remote_in6.s6_addr[c] & mask_in6.s6_addr[c];
      if (c1 ^ c2) {
        same = false;
        break;
      }
    }
    // At last, we need to compare scope id
    // Two Link-type addresses can have the same subnet address even though they are not in the same scope
    // For Global type, this field is 0, so a comparison wouldn't matter
    same &= (local_addr->sin6_scope_id == remote_addr.sin6_scope_id);
    return same;
  } else {
    WARN("Net : Unsupported address family type");
    return false;
  }
}

int ncclFindInterfaceMatchSubnet(char* ifNames, union ncclSocketAddress* localAddrs, union ncclSocketAddress* remoteAddr, int ifNameMaxSize, int maxIfs) {
#ifdef ENABLE_TRACE
  char line[SOCKET_NAME_MAXLEN+1];
#endif
  char line_a[SOCKET_NAME_MAXLEN+1];
  int found = 0;
  struct ifaddrs *interfaces, *interface;
  getifaddrs(&interfaces);
  for (interface = interfaces; interface && !found; interface = interface->ifa_next) {
    if (interface->ifa_addr == NULL) continue;

    /* We only support IPv4 & IPv6 */
    int family = interface->ifa_addr->sa_family;
    if (family != AF_INET && family != AF_INET6)
      continue;

    // check against user specified interfaces
    if (!matchSubnet(*interface, remoteAddr)) {
      continue;
    }

    // Store the local IP address
    int salen = (family == AF_INET) ? sizeof(struct sockaddr_in) : sizeof(struct sockaddr_in6);
    memcpy(localAddrs+found, interface->ifa_addr, salen);

    // Store the interface name
    strncpy(ifNames+found*ifNameMaxSize, interface->ifa_name, ifNameMaxSize);

    TRACE(NCCL_INIT|NCCL_NET,"NET : Found interface %s:%s in the same subnet as remote address %s", interface->ifa_name, ncclSocketToString(localAddrs+found, line), ncclSocketToString(remoteAddr, line_a));
    found++;
    if (found == maxIfs) break;
  }

  if (found == 0) {
    WARN("Net : No interface found in the same subnet as remote address %s", ncclSocketToString(remoteAddr, line_a));
  }
  freeifaddrs(interfaces);
  return found;
}

ncclResult_t ncclSocketGetAddrFromString(union ncclSocketAddress* ua, const char* ip_port_pair) {
  if (!(ip_port_pair && strlen(ip_port_pair) > 1)) {
    WARN("Net : string is null");
    return ncclInvalidArgument;
  }

  bool ipv6 = ip_port_pair[0] == '[';
  /* Construct the sockaddress structure */
  if (!ipv6) {
    struct netIf ni;
    // parse <ip_or_hostname>:<port> string, expect one pair
    if (parseStringList(ip_port_pair, &ni, 1) != 1) {
      WARN("Net : No valid <IPv4_or_hostname>:<port> pair found");
      return ncclInvalidArgument;
    }

    struct addrinfo hints, *p;
    int rv;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    if ( (rv = getaddrinfo(ni.prefix, NULL, &hints, &p)) != 0) {
      WARN("Net : error encountered when getting address info : %s", gai_strerror(rv));
      return ncclInvalidArgument;
    }

    // use the first
    if (p->ai_family == AF_INET) {
      struct sockaddr_in& sin = ua->sin;
      memcpy(&sin, p->ai_addr, sizeof(struct sockaddr_in));
      sin.sin_family = AF_INET;                        // IPv4
      //inet_pton(AF_INET, ni.prefix, &(sin.sin_addr));  // IP address
      sin.sin_port = htons(ni.port);                   // port
    } else if (p->ai_family == AF_INET6) {
      struct sockaddr_in6& sin6 = ua->sin6;
      memcpy(&sin6, p->ai_addr, sizeof(struct sockaddr_in6));
      sin6.sin6_family = AF_INET6;                     // IPv6
      sin6.sin6_port = htons(ni.port);                 // port
      sin6.sin6_flowinfo = 0;                          // needed by IPv6, but possibly obsolete
      sin6.sin6_scope_id = 0;                          // should be global scope, set to 0
    } else {
      WARN("Net : unsupported IP family");
      freeaddrinfo(p);
      return ncclInvalidArgument;
    }

    freeaddrinfo(p); // all done with this structure

  } else {
    int i, j = -1, len = strlen(ip_port_pair);
    for (i = 1; i < len; i++) {
      if (ip_port_pair[i] == '%') j = i;
      if (ip_port_pair[i] == ']') break;
    }
    if (i == len) {
      WARN("Net : No valid [IPv6]:port pair found");
      return ncclInvalidArgument;
    }
    bool global_scope = (j == -1 ? true : false);     // If no % found, global scope; otherwise, link scope

    char ip_str[NI_MAXHOST], port_str[NI_MAXSERV], if_name[IFNAMSIZ];
    memset(ip_str, '\0', sizeof(ip_str));
    memset(port_str, '\0', sizeof(port_str));
    memset(if_name, '\0', sizeof(if_name));
    strncpy(ip_str, ip_port_pair+1, global_scope ? i-1 : j-1);
    strncpy(port_str, ip_port_pair+i+2, len-i-1);
    int port = atoi(port_str);
    if (!global_scope) strncpy(if_name, ip_port_pair+j+1, i-j-1); // If not global scope, we need the intf name

    struct sockaddr_in6& sin6 = ua->sin6;
    sin6.sin6_family = AF_INET6;                       // IPv6
    inet_pton(AF_INET6, ip_str, &(sin6.sin6_addr));    // IP address
    sin6.sin6_port = htons(port);                      // port
    sin6.sin6_flowinfo = 0;                            // needed by IPv6, but possibly obsolete
    sin6.sin6_scope_id = global_scope ? 0 : if_nametoindex(if_name);  // 0 if global scope; intf index if link scope
  }
  return ncclSuccess;
}
//该函数用于查找并返回可用的网络接口,优先顺序为：用户指定接口 > InfiniBand（ib）> 与NCCL_COMM_ID同一子网的接口 > 非docker/lo的其他接口 > docker > lo。
int ncclFindInterfaces(char* ifNames, union ncclSocketAddress *ifAddrs, int ifNameMaxSize, int maxIfs) {
  static int shownIfName = 0;
  int nIfs = 0;
  // Allow user to force the INET socket family selection 从环境变量中获取socket的地址族，目前是选择ipv4或ipv6。
  int sock_family = envSocketFamily();
  // User specified interface 指定的网络接口
  const char* env = ncclGetEnv("NCCL_SOCKET_IFNAME");
  if (env && strlen(env) > 1) {
    INFO(NCCL_ENV, "NCCL_SOCKET_IFNAME set by environment to %s", env);
    // Specified by user : find or fail
    if (shownIfName++ == 0) INFO(NCCL_NET, "NCCL_SOCKET_IFNAME set to %s", env);
    nIfs = findInterfaces(env, ifNames, ifAddrs, sock_family, ifNameMaxSize, maxIfs);
  } else {
    // Try to automatically pick the right one
    // Start with IB
    nIfs = findInterfaces("ib", ifNames, ifAddrs, sock_family, ifNameMaxSize, maxIfs);
    // else see if we can get some hint from COMM ID
    if (nIfs == 0) {
      const char* commId = ncclGetEnv("NCCL_COMM_ID");
      if (commId && strlen(commId) > 1) {
        INFO(NCCL_ENV, "NCCL_COMM_ID set by environment to %s", commId);
        // Try to find interface that is in the same subnet as the IP in comm id
        union ncclSocketAddress idAddr;
        ncclSocketGetAddrFromString(&idAddr, commId);
        nIfs = ncclFindInterfaceMatchSubnet(ifNames, ifAddrs, &idAddr, ifNameMaxSize, maxIfs);
      }
    }
    // Then look for anything else (but not docker or lo)
    if (nIfs == 0) nIfs = findInterfaces("^docker,lo", ifNames, ifAddrs, sock_family, ifNameMaxSize, maxIfs);
    // Finally look for docker, then lo.
    if (nIfs == 0) nIfs = findInterfaces("docker", ifNames, ifAddrs, sock_family, ifNameMaxSize, maxIfs);
    if (nIfs == 0) nIfs = findInterfaces("lo", ifNames, ifAddrs, sock_family, ifNameMaxSize, maxIfs);
  }
  return nIfs;
}
//根据sock的地址和fd进行监听（bind+listen），这个过程中进行了细粒度的套接字设置（可重用等）
ncclResult_t ncclSocketListen(struct ncclSocket* sock) {
  if (sock == NULL) {
    WARN("ncclSocketListen: pass NULL socket");
    return ncclInvalidArgument;
  }
  if (sock->fd == -1) {
    WARN("ncclSocketListen: file descriptor is -1");
    return ncclInvalidArgument;
  }

  if (socketToPort(&sock->addr)) {//如果端口号不为0
    // Port is forced by env. Make sure we get the port.
    int opt = 1;
    //setsockopt 是 ​​套接字编程​​ 中用于设置套接字选项（Socket Options）的核心函数，允许开发者精细控制套接字的行为（如超时、缓冲区大小、重用地址等）
    // SOL_SOCKET是通用套接字选项（与协议无关）、还可以选择tcp、IP层。 SO_REUSEADDR选项表示允许重用本地地址和端口。
    //SO_REUSEADDR 允许你的程序​​立即重用​​之前被占用（比如程序崩溃后还没释放）的 ​​IP地址 + 端口​​，避免出现“地址已在使用”的错误。
    //同一个 IP + 端口 组合只能被 ​​一个 TCP 套接字​​ 绑定。
    //一般服务器的监听socket都应该打开它。需要在bind之前设定
    SYSCHECK(setsockopt(sock->fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)), "setsockopt");
#if defined(SO_REUSEPORT)
    // SO_REUSEPORT 允许多个套接字绑定到同一个 IP + 端口 组合(多进程的情况)。多个进程同时监听同一端口，内核自动分配连接请求（替代传统的单进程 accept 模型）。
    SYSCHECK(setsockopt(sock->fd, SOL_SOCKET, SO_REUSEPORT, &opt, sizeof(opt)), "setsockopt");
#endif
  }

  // addr port should be 0 (Any port) 如果地址中的 port 为 0，则系统会自动分配一个未使用的端口。
  SYSCHECK(bind(sock->fd, &sock->addr.sa, sock->salen), "bind");

  /* Get the assigned Port */
  socklen_t size = sock->salen;
  //获取系统实际分配的端口号（因为 bind 时可能用了 0 端口，由系统自动分配）。
  SYSCHECK(getsockname(sock->fd, &sock->addr.sa, &size), "getsockname");

#ifdef ENABLE_TRACE
  char line[SOCKET_NAME_MAXLEN+1];
  TRACE(NCCL_INIT|NCCL_NET,"Listening on socket %s", ncclSocketToString(&sock->addr, line));
#endif

  /* Put the socket in listen mode
   * NB: The backlog will be silently truncated to the value in /proc/sys/net/core/somaxconn
   */
  //将 socket 设置为监听状态，准备接受客户端连接.这里设为 16384，表示最多允许 16384 个连接排队等待 accept。
  //实际值可能会被系统限制 /proc/sys/net/core/somaxconn 截断
  SYSCHECK(listen(sock->fd, 16384), "listen");
  sock->state = ncclSocketStateReady;
  return ncclSuccess;
}
//把sock里面地址拷贝到addr中
ncclResult_t ncclSocketGetAddr(struct ncclSocket* sock, union ncclSocketAddress* addr) {
  if (sock == NULL) {
    WARN("ncclSocketGetAddr: pass NULL socket");
    return ncclInvalidArgument;
  }
  if (sock->state != ncclSocketStateReady) return ncclInternalError;
  memcpy(addr, &sock->addr, sizeof(union ncclSocketAddress));
  return ncclSuccess;
}
//
static ncclResult_t socketTryAccept(struct ncclSocket* sock) {
  socklen_t socklen = sizeof(union ncclSocketAddress);
  sock->fd = accept(sock->acceptFd, (struct sockaddr*)&sock->addr, &socklen);
  if (sock->fd != -1) {
    sock->state = ncclSocketStateAccepted;
  } else if (errno == ENETDOWN || errno == EPROTO || errno == ENOPROTOOPT || errno == EHOSTDOWN ||
             errno == ENONET || errno == EHOSTUNREACH || errno == EOPNOTSUPP || errno == ENETUNREACH) {
    /* per accept's man page, for linux sockets, the following errors might be already pending errors
     * and should be considered as EAGAIN. To avoid infinite loop in case of errors, we use the retry count*/
    if (++sock->errorRetries == ncclParamRetryCnt()) {
      WARN("socketTryAccept: exceeded error retry count (%d), %s", sock->errorRetries, strerror(errno));
      return ncclSystemError;
    }
    INFO(NCCL_ALL, "Call to accept returned %s, retrying", strerror(errno));
  } else if (errno != EAGAIN && errno != EWOULDBLOCK) {
    WARN("socketTryAccept: Accept failed: %s", strerror(errno));
    return ncclSystemError;
  }
  return ncclSuccess;
}

static ncclResult_t socketSetFlags(struct ncclSocket* sock) {
  const int one = 1;
  /* Set socket as non-blocking if async or if we need to be able to abort */
  if ((sock->asyncFlag || sock->abortFlag) && sock->fd >= 0) {
    int flags;
    SYSCHECK(flags = fcntl(sock->fd, F_GETFL), "fcntl");
    SYSCHECK(fcntl(sock->fd, F_SETFL, flags | O_NONBLOCK), "fcntl");
  }
  SYSCHECK(setsockopt(sock->fd, IPPROTO_TCP, TCP_NODELAY, (char*)&one, sizeof(int)), "setsockopt");
  return ncclSuccess;
}

static ncclResult_t socketFinalizeAccept(struct ncclSocket* sock) {
  uint64_t magic;
  enum ncclSocketType type;
  int received;
  // once accepted, linux sockets do NOT inherit file status flags such as O_NONBLOCK (BSD ones do)
  NCCLCHECK(socketSetFlags(sock));

  if (sock->asyncFlag == 0 || sock->finalizeCounter < sizeof(magic)) {
    if (sock->asyncFlag == 0) {
      received = 0;
      NCCLCHECK(socketWait(NCCL_SOCKET_RECV, sock, &magic, sizeof(magic), &received));
    } else {
      received = sock->finalizeCounter;
      NCCLCHECK(socketProgress(NCCL_SOCKET_RECV, sock, sock->finalizeBuffer, sizeof(magic), &received));
      sock->finalizeCounter = received;
      if (received < sizeof(magic)) return ncclSuccess;
      memcpy(&magic, sock->finalizeBuffer, sizeof(magic));
    }
    if (magic != sock->magic) {
      WARN("socketFinalizeAccept: wrong magic %lx != %lx", magic, sock->magic);
      close(sock->fd);
      sock->fd = -1;
      // Ignore spurious connection and accept again
      sock->state = ncclSocketStateAccepting;
      return ncclSuccess;
    }
  }
  if (sock->asyncFlag == 0) {
    received = 0;
    NCCLCHECK(socketWait(NCCL_SOCKET_RECV, sock, &type, sizeof(type), &received));
  } else {
    received = sock->finalizeCounter - sizeof(magic);
    NCCLCHECK(socketProgress(NCCL_SOCKET_RECV, sock, sock->finalizeBuffer, sizeof(type), &received));
    sock->finalizeCounter = received + sizeof(magic);
    if (received < sizeof(type)) return ncclSuccess;
    memcpy(&type, sock->finalizeBuffer, sizeof(type));
  }
  if (type != sock->type) {
    WARN("socketFinalizeAccept: wrong type %d != %d", type, sock->type);
    sock->state = ncclSocketStateError;
    close(sock->fd);
    sock->fd = -1;
    return ncclInternalError;
  } else {
    sock->state = ncclSocketStateReady;
  }
  return ncclSuccess;
}
//重置socket的文件描述符
static ncclResult_t socketResetFd(struct ncclSocket* sock) {
  ncclResult_t ret = ncclSuccess;
  int fd = -1;
  SYSCHECKGOTO(fd = socket(sock->addr.sa.sa_family, SOCK_STREAM, 0), "socket", ret, cleanup);
  // if sock->fd is valid, close it and reuse its number
  if (sock->fd != -1) {
    SYSCHECKGOTO(dup2(fd, sock->fd), "dup2", ret, cleanup);// 
    SYSCHECKGOTO(close(fd), "close", ret, cleanup);// 
  } else {
    sock->fd = fd;
  }
  NCCLCHECKGOTO(socketSetFlags(sock), ret, exit);
exit:
  return ret;
cleanup:
  // cleanup fd, leave sock->fd untouched
  if (fd != -1) {
    (void)close(fd);
  }
  goto exit;
}
static ncclResult_t socketConnectCheck(struct ncclSocket* sock, int errCode, const char funcName[]) {
  if (errCode == 0) {
    sock->state = ncclSocketStateConnected;
  } else if (errCode == EINPROGRESS) {
    sock->state = ncclSocketStateConnectPolling;
  } else if (errCode == ETIMEDOUT || errCode == EHOSTUNREACH || errCode == ECONNREFUSED) {
    if (sock->customRetry == 0) {
      if (sock->errorRetries++ == ncclParamRetryCnt()) {
        sock->state = ncclSocketStateError;
        WARN("%s: connect returned %s, exceeded error retry count (%d)", funcName, strerror(errCode), sock->errorRetries);
        return ncclRemoteError;
      }
      unsigned int sleepTime = sock->errorRetries * ncclParamRetryTimeOut();
      INFO(NCCL_ALL, "%s: connect returned %s, retrying (%d/%ld) after sleep for %u msec", funcName, strerror(errCode), sock->errorRetries, ncclParamRetryCnt(), sleepTime);
      msleep(sleepTime);
    }
    NCCLCHECK(socketResetFd(sock)); /* in case of failure in connect, socket state is unspecified */
    sock->state = ncclSocketStateConnecting;
  } else {
    char line[SOCKET_NAME_MAXLEN+1];
    sock->state = ncclSocketStateError;
    WARN("%s: Connect to %s failed : %s", funcName, ncclSocketToString(&sock->addr, line), strerror(errCode));
    return ncclSystemError;
  }
  return ncclSuccess;
}
static ncclResult_t socketStartConnect(struct ncclSocket* sock) {
  /* blocking/non-blocking connect() is determined by asyncFlag. */
  int ret = connect(sock->fd, &sock->addr.sa, sock->salen);
  return socketConnectCheck(sock, (ret == -1) ? errno : 0, __func__);
}

static ncclResult_t socketPollConnect(struct ncclSocket* sock) {
  struct pollfd pfd;
  int timeout = 1, ret;
  socklen_t rlen = sizeof(int);

  memset(&pfd, 0, sizeof(struct pollfd));
  pfd.fd = sock->fd;
  pfd.events = POLLOUT;
  ret = poll(&pfd, 1, timeout);

  if (ret == 0 || (ret < 0 && errno == EINTR)) {
    return ncclSuccess;
  } else if (ret < 0) {
    WARN("socketPollConnect poll() failed with error %s", strerror(errno));
    return ncclRemoteError;
  } else if (ret != 1 || (pfd.revents & POLLOUT) == 0) {
    WARN("socketPollConnect poll() returned %d%s", ret, (pfd.revents & POLLOUT) ? "" : ", no POLLOUT events");
    return ncclSystemError;
  }

  /* check socket status */
  SYSCHECK(getsockopt(sock->fd, SOL_SOCKET, SO_ERROR, (void*)&ret, &rlen), "getsockopt");
  return socketConnectCheck(sock, ret, __func__);
}

ncclResult_t ncclSocketPollConnect(struct ncclSocket* sock) {
  if (sock == NULL) {
    WARN("ncclSocketPollConnect: pass NULL socket");
    return ncclInvalidArgument;
  }
  NCCLCHECK(socketPollConnect(sock));
  return ncclSuccess;
}
//如果sock->asyncFlag为0，则发送magic和type到对端，表示连接的完成；如果为1，则使用异步方式发送，直到发送完magic和type。
static ncclResult_t socketFinalizeConnect(struct ncclSocket* sock) {
  int sent;//已发送的数据量
  //（同步模式），则直接通过 socketWait 发送 magic 和 type，确保对端收到正确的标识符。(magic表示正确的连接，type表示正确的socket类型（比如是用于bootstrap、proxy还是等等）)
  if (sock->asyncFlag == 0) {
    sent = 0;
    NCCLCHECK(socketWait(NCCL_SOCKET_SEND, sock, &sock->magic, sizeof(sock->magic), &sent));
    sent = 0;
    NCCLCHECK(socketWait(NCCL_SOCKET_SEND, sock, &sock->type, sizeof(sock->type), &sent));
  } else {//异步模式，则分阶段发送数据，直到所有数据发送完毕。
    
    if (sock->finalizeCounter < sizeof(sock->magic)) {
      // 发送 magic 的部分数据
      sent = sock->finalizeCounter;
      NCCLCHECK(socketProgress(NCCL_SOCKET_SEND, sock, &sock->magic, sizeof(sock->magic), &sent));
      sock->finalizeCounter = sent;
      if (sent < sizeof(sock->magic)) return ncclSuccess;//如果未完成，返回 ncclSuccess 等待下次调用
    }
    //此时finalizeCounter就是已经发送的数据大小，所以这里就接着发type的数据
    sent = sock->finalizeCounter - sizeof(sock->magic);
    NCCLCHECK(socketProgress(NCCL_SOCKET_SEND, sock, &sock->type, sizeof(sock->type), &sent));
    sock->finalizeCounter = sent + sizeof(sock->magic);
    if (sent < sizeof(sock->type)) return ncclSuccess;//如果未完成，返回 ncclSuccess 等待下次调用（因为返回了ncclSuccess，所以不会修改socket的state）
  }
  //最终结束了，将socket状态设置为ncclSocketStateReady
  sock->state = ncclSocketStateReady;
  return ncclSuccess;
}
//处理socket的各个状态转换
static ncclResult_t socketProgressState(struct ncclSocket* sock) {
  if (sock->state == ncclSocketStateAccepting) {
    NCCLCHECK(socketTryAccept(sock));
  }
  if (sock->state == ncclSocketStateAccepted) {
    NCCLCHECK(socketFinalizeAccept(sock));
  }
  if (sock->state == ncclSocketStateConnecting) {
    NCCLCHECK(socketStartConnect(sock));
  }
  if (sock->state == ncclSocketStateConnectPolling) {
    NCCLCHECK(socketPollConnect(sock));
  }
  if (sock->state == ncclSocketStateConnected) {
    NCCLCHECK(socketFinalizeConnect(sock));
  }
  return ncclSuccess;
}

ncclResult_t ncclSocketReady(struct ncclSocket* sock, int *running) {
  if (sock == NULL) {
    *running = 0;
    return ncclSuccess;
  }
  if (sock->state == ncclSocketStateError || sock->state == ncclSocketStateClosed) {
    WARN("ncclSocketReady: unexpected socket state %d", sock->state);
    return ncclRemoteError;
  }
  *running = (sock->state == ncclSocketStateReady) ? 1 : 0;
  if (*running == 0) {
    NCCLCHECK(socketProgressState(sock));
    *running = (sock->state == ncclSocketStateReady) ? 1 : 0;
  }
  return ncclSuccess;
}

ncclResult_t ncclSocketConnect(struct ncclSocket* sock) {
#ifdef ENABLE_TRACE
  char line[SOCKET_NAME_MAXLEN+1];
#endif

  if (sock == NULL) {
    WARN("ncclSocketConnect: pass NULL socket");
    return ncclInvalidArgument;
  }
  if (sock->fd == -1) {
    WARN("ncclSocketConnect: file descriptor is -1");
    return ncclInvalidArgument;
  }

  if (sock->state != ncclSocketStateInitialized) {
    WARN("ncclSocketConnect: wrong socket state %d", sock->state);
    if (sock->state == ncclSocketStateError) return ncclRemoteError;
    return ncclInternalError;
  }
  TRACE(NCCL_INIT|NCCL_NET,"Connecting to socket %s", ncclSocketToString(&sock->addr, line));

  sock->state = ncclSocketStateConnecting;
  sock->finalizeCounter = 0;
  do {
    NCCLCHECK(socketProgressState(sock));//直到ready状态为止
  } while (sock->asyncFlag == 0 &&
      (sock->abortFlag == NULL || __atomic_load_n(sock->abortFlag, __ATOMIC_ACQUIRE) == 0) &&
      (sock->state == ncclSocketStateConnecting ||
       sock->state == ncclSocketStateConnectPolling ||
       sock->state == ncclSocketStateConnected));

  if (sock->abortFlag && __atomic_load_n(sock->abortFlag, __ATOMIC_ACQUIRE)) return ncclInternalError;

  switch (sock->state) {
    case ncclSocketStateConnecting:
    case ncclSocketStateConnectPolling:
    case ncclSocketStateConnected:
    case ncclSocketStateReady:
      return ncclSuccess;
    case ncclSocketStateError:
      return ncclSystemError;
    default:
      WARN("ncclSocketConnect: wrong socket state %d", sock->state);
      return ncclInternalError;
  }
}
//nccl封装的accept函数，实现了标准socket accept操作的封装，并添加了状态管理和错误处理
// accept到的信息放在sock中
ncclResult_t ncclSocketAccept(struct ncclSocket* sock, struct ncclSocket* listenSock) {
  ncclResult_t ret = ncclSuccess;

  if (listenSock == NULL || sock == NULL) {
    WARN("ncclSocketAccept: pass NULL socket");
    ret = ncclInvalidArgument;
    goto exit;
  }
  if (listenSock->state != ncclSocketStateReady) {//listen函数里面有设定这个，表明已经调用了listen函数。
    WARN("ncclSocketAccept: wrong socket state %d", listenSock->state);
    if (listenSock->state == ncclSocketStateError)
      ret = ncclSystemError;
    else
      ret = ncclInternalError;
    goto exit;
  }

  if (sock->acceptFd == -1) {//还没初始化，先初始化一下。
    memcpy(sock, listenSock, sizeof(struct ncclSocket));
    sock->acceptFd = listenSock->fd;// 这里的acceptFd是指监听套接字的文件描述符
    sock->state = ncclSocketStateAccepting;
    sock->finalizeCounter = 0;
  }
//处理接受连接的过程。
  do {
    NCCLCHECKGOTO(socketProgressState(sock), ret, exit);// 推进socket状态机，这里由于是出于accepting状态，其实相当于调用accept函数
  } while (sock->asyncFlag == 0 &&
      (sock->abortFlag == NULL || __atomic_load_n(sock->abortFlag, __ATOMIC_ACQUIRE) == 0) &&
      (sock->state == ncclSocketStateAccepting ||
       sock->state == ncclSocketStateAccepted));// 同步模式下

  if (sock->abortFlag && __atomic_load_n(sock->abortFlag, __ATOMIC_ACQUIRE)) return ncclInternalError;

  switch (sock->state) {
    case ncclSocketStateAccepting:
    case ncclSocketStateAccepted:
    case ncclSocketStateReady:
      ret = ncclSuccess;
      break;
    case ncclSocketStateError:
      ret = ncclSystemError;
      break;
    default:
      WARN("ncclSocketAccept: wrong socket state %d", sock->state);
      ret = ncclInternalError;
      break;
  }

exit:
  return ret;
}
//初始化一个socket，默认是阻塞的
ncclResult_t ncclSocketInit(struct ncclSocket* sock, const union ncclSocketAddress* addr, uint64_t magic, enum ncclSocketType type, volatile uint32_t* abortFlag, int asyncFlag, int customRetry) {
  ncclResult_t ret = ncclSuccess;

  if (sock == NULL) goto exit;
  sock->errorRetries = 0;
  sock->abortFlag = abortFlag;
  sock->asyncFlag = asyncFlag;
  sock->state = ncclSocketStateInitialized;
  sock->magic = magic;
  sock->type = type;
  sock->fd = -1;
  sock->acceptFd = -1;
  sock->customRetry = customRetry;

  if (addr) {
    /* IPv4/IPv6 support */
    int family;
    memcpy(&sock->addr, addr, sizeof(union ncclSocketAddress));
    family = sock->addr.sa.sa_family;
    if (family != AF_INET && family != AF_INET6) {
      char line[SOCKET_NAME_MAXLEN+1];
      WARN("ncclSocketInit: connecting to address %s with family %d is neither AF_INET(%d) nor AF_INET6(%d)",
          ncclSocketToString(&sock->addr, line), family, AF_INET, AF_INET6);
      ret = ncclInternalError;
      goto exit;
    }
    sock->salen = (family == AF_INET) ? sizeof(struct sockaddr_in) : sizeof(struct sockaddr_in6);
    // in case of error, we close the fd before returning as it's unclear if the caller has to use ncclSocketClose for cleanup
    NCCLCHECKGOTO(socketResetFd(sock), ret, fail);
  } else {
    memset(&sock->addr, 0, sizeof(union ncclSocketAddress));
  }
exit:
  return ret;
fail:
  if (sock->fd != -1) {
    close(sock->fd);
    sock->fd = -1;
  }
  goto exit;
}
//NCCL中用于处理套接字非阻塞I/O操作的关键函数。这个函数是对内部 socketProgress 函数的公开封装，提供了错误检查和参数验证功能。
//尝试发送或接收数据，但不会阻塞等待操作完成
ncclResult_t ncclSocketProgress(int op, struct ncclSocket* sock, void* ptr, int size, int* offset, int* closed) {
  if (sock == NULL) {
    WARN("ncclSocketProgress: pass NULL socket");
    return ncclInvalidArgument;
  }
  NCCLCHECK(socketProgress(op, sock, ptr, size, offset, closed));
  return ncclSuccess;
}

ncclResult_t ncclSocketWait(int op, struct ncclSocket* sock, void* ptr, int size, int* offset) {
  if (sock == NULL) {
    WARN("ncclSocketWait: pass NULL socket");
    return ncclInvalidArgument;
  }
  NCCLCHECK(socketWait(op, sock, ptr, size, offset));
  return ncclSuccess;
}

ncclResult_t ncclSocketSend(struct ncclSocket* sock, void* ptr, int size) {
  int offset = 0;
  if (sock == NULL) {
    WARN("ncclSocketSend: pass NULL socket");
    return ncclInvalidArgument;
  }
  if (sock->state != ncclSocketStateReady) {
    WARN("ncclSocketSend: socket state (%d) is not ready", sock->state);
    return ncclInternalError;
  }
  NCCLCHECK(socketWait(NCCL_SOCKET_SEND, sock, ptr, size, &offset));
  return ncclSuccess;
}

ncclResult_t ncclSocketRecv(struct ncclSocket* sock, void* ptr, int size) {
  int offset = 0;
  if (sock == NULL) {
    WARN("ncclSocketRecv: pass NULL socket");
    return ncclInvalidArgument;
  }
  if (sock->state != ncclSocketStateReady && sock->state != ncclSocketStateTerminating) {
    WARN("ncclSocketRecv: socket state (%d) is not ready", sock->state);
    return ncclInternalError;
  }
  NCCLCHECK(socketWait(NCCL_SOCKET_RECV, sock, ptr, size, &offset));
  return ncclSuccess;
}

ncclResult_t ncclSocketSendRecv(struct ncclSocket* sendSock, void* sendPtr, int sendSize, struct ncclSocket* recvSock, void* recvPtr, int recvSize) {
  int sendOffset = 0, recvOffset = 0;
  if (sendSock == NULL || recvSock == NULL) {
    WARN("ncclSocketSendRecv: invalid socket %p/%p", sendSock, recvSock);
    return ncclInternalError;
  }
  if (sendSock->state != ncclSocketStateReady ||
      (recvSock->state != ncclSocketStateReady && recvSock->state != ncclSocketStateTerminating)) {
    WARN("ncclSocketSendRecv: socket state (%d/%d) is not ready", sendSock->state, recvSock->state);
    return ncclInternalError;
  }
  while (sendOffset < sendSize || recvOffset < recvSize) {
    if (sendOffset < sendSize) NCCLCHECK(socketProgress(NCCL_SOCKET_SEND, sendSock, sendPtr, sendSize, &sendOffset));
    if (recvOffset < recvSize) NCCLCHECK(socketProgress(NCCL_SOCKET_RECV, recvSock, recvPtr, recvSize, &recvOffset));
  }
  return ncclSuccess;
}


// Receive or detect connection closed
ncclResult_t ncclSocketTryRecv(struct ncclSocket* sock, void* ptr, int size, int* closed, bool blocking) {
  int offset = 0;
  if (sock == NULL) {
    WARN("ncclSocketTryRecv: pass NULL socket");
    return ncclInvalidArgument;
  }
  *closed = 0;
  // Block until connection closes or nbytes received
  if (blocking) {
    while (offset < size) {
      NCCLCHECK(socketProgressOpt(NCCL_SOCKET_RECV, sock, ptr, size, &offset, 0, closed));
      if (*closed) return ncclSuccess;
    }
  } else {
    NCCLCHECK(socketProgressOpt(NCCL_SOCKET_RECV, sock, ptr, size, &offset, 0, closed));
    if (*closed) return ncclSuccess;

    // If any bytes were received, block waiting for the rest
    if (offset > 0) {
      while (offset < size) {
        NCCLCHECK(socketProgressOpt(NCCL_SOCKET_RECV, sock, ptr, size, &offset, 0, closed));
        if (*closed) return ncclSuccess;
      }
    // No bytes were received, return ncclInProgress
    } else {
      return ncclInProgress;
    }
  }
  return ncclSuccess;
}

// Make it possible to close just one part of a socket.
ncclResult_t ncclSocketShutdown(struct ncclSocket* sock, int how) {
  if (sock != NULL) {
    if (sock->fd >= 0) {
      shutdown(sock->fd, how);
    }
    sock->state = ncclSocketStateTerminating;
  }
  return ncclSuccess;
}

ncclResult_t ncclSocketClose(struct ncclSocket* sock) {
  if (sock != NULL) {
    if (sock->state > ncclSocketStateNone && sock->state < ncclSocketStateNum && sock->fd >= 0) {
      /* shutdown() is needed to send FIN packet to proxy thread; shutdown() is not affected
       * by refcount of fd, but close() is. close() won't close a fd and send FIN packet if
       * the fd is duplicated (e.g. fork()). So shutdown() guarantees the correct and graceful
       * connection close here. */
      shutdown(sock->fd, SHUT_RDWR);
      close(sock->fd);
    }
    sock->state = ncclSocketStateClosed;
    sock->fd = -1;
  }
  return ncclSuccess;
}
//获取socket的fd
ncclResult_t ncclSocketGetFd(struct ncclSocket* sock, int* fd) {
  if (sock == NULL) {
    WARN("ncclSocketGetFd: pass NULL socket");
    return ncclInvalidArgument;
  }
  if (fd) *fd = sock->fd;
  return ncclSuccess;
}

ncclResult_t ncclSocketSetFd(int fd, struct ncclSocket* sock) {
  if (sock == NULL) {
    WARN("ncclSocketGetFd: pass NULL socket");
    return ncclInvalidArgument;
  }
  sock->fd = fd;
  return ncclSuccess;
}
