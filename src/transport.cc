/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "info.h"
#include "bootstrap.h"
#define ENABLE_TIMER 0
#include "timer.h"
#include "transport.h"

struct ncclTransport* ncclTransports[NTRANSPORTS] = {
  &p2pTransport,
  &shmTransport,
  &netTransport,
  &collNetTransport
};
//为两个通信节点之间选择并设置合适的传输方式，就是选p2p、shm、net、collNet之一
template <int type>
static ncclResult_t selectTransport(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclConnect* connect, int channelId, int peer, int connIndex, int* transportType) {
  struct ncclPeerInfo* myInfo = comm->peerInfo+comm->rank;
  struct ncclPeerInfo* peerInfo = comm->peerInfo+peer;
  //获取对应的connector，type=0表示获取peer（从哪个接收的）的connector
  struct ncclConnector* connector = (type == 1) ? comm->channels[channelId].peers[peer]->send + connIndex :
                                                  comm->channels[channelId].peers[peer]->recv + connIndex;
  for (int t=0; t<NTRANSPORTS; t++) {//
    struct ncclTransport *transport = ncclTransports[t];
    struct ncclTransportComm* transportComm = type == 1 ? &transport->send : &transport->recv;
    int ret = 0;
    NCCLCHECK(transport->canConnect(&ret, comm, graph, myInfo, peerInfo));//若可以建立连接
    if (ret) {
      connector->transportComm = transportComm;//若某传输方式支持连接，则设置该传输方式,并调用其 setup 方法进行初始化；
      NCCLCHECK(transportComm->setup(comm, graph, myInfo, peerInfo, connect, connector, channelId, connIndex));
      if (transportType) *transportType = t;
      return ncclSuccess;
    }
  }
  WARN("No transport found for rank %d[%lx] -> rank %d[%lx]", myInfo->rank, myInfo->busId, peerInfo->rank, peerInfo->busId);
  return ncclSystemError;
}
//名字叫connect，其实是设置一下comm->connectRecv和comm->connectSend的掩码信息
//后续可以根据这个掩码信息判断哪些peer之间是存在通道的
ncclResult_t ncclTransportP2pConnect(struct ncclComm* comm, int channelId, int nrecv, int* peerRecv, int nsend, int* peerSend, int connIndex) {
  TRACE(NCCL_INIT, "nsend %d nrecv %d", nsend, nrecv);
  struct ncclChannel* channel = &comm->channels[channelId];
  uint64_t mask = 1UL << channel->id;//设置当前channel的标志位，其中只有对应于当前通道 ID 的位被置为 1
  for (int i=0; i<nrecv; i++) {//遍历每个pre peer（从peerRecv接收信息）
    int peer = peerRecv[i];
    if (peer == -1 || peer >= comm->nRanks || peer == comm->rank || channel->peers[peer]->recv[connIndex].connected) continue;
    comm->connectRecv[peer] |= mask;//说明当前rank从peer处接收信息，并且该通道已经连接。mask就是为了标记当前rank到peerrank的channel的使用情况
  }
  for (int i=0; i<nsend; i++) {
    int peer = peerSend[i];
    if (peer == -1 || peer >= comm->nRanks || peer == comm->rank || channel->peers[peer]->send[connIndex].connected) continue;
    comm->connectSend[peer] |= mask;
  }
  return ncclSuccess;
}

void dumpData(struct ncclConnect* data, int ndata) {
  for (int n=0; n<ndata; n++) {
    printf("[%d] ", n);
    uint8_t* d = (uint8_t*)data;
    for (int i=0; i<sizeof(struct ncclConnect); i++) printf("%02x", d[i]);
    printf("\n");
  }
}

NCCL_PARAM(ConnectRoundMaxPeers, "CONNECT_ROUND_MAX_PEERS", 128);
NCCL_PARAM(ReportConnectProgress, "REPORT_CONNECT_PROGRESS", 0);
#include <sys/time.h>

ncclResult_t ncclTransportCheckP2pType(struct ncclComm* comm, bool* intraNodeP2pSupport, bool* directMode) {
  bool supportFlag = true;
  bool directFlag = false;
  if (comm->localRanks == 1) {
    supportFlag = false;
  } else {
    for (int i = 0; i < comm->localRanks; ++i) {
      for (int j = i + 1; j < comm->localRanks; ++j) {
        int ipeer = comm->localRankToRank[i];
        int jpeer = comm->localRankToRank[j];
        struct ncclPeerInfo* ipeerInfo = &comm->peerInfo[ipeer];
        struct ncclPeerInfo* jpeerInfo = &comm->peerInfo[jpeer];
        int canConnect = 0;
        //是否可以连接
        NCCLCHECK(ncclTransports[0]->canConnect(&canConnect, comm, NULL, ipeerInfo, jpeerInfo));
        if (!canConnect && supportFlag == true) {
          supportFlag = false;
        }
        //如果两个 peer 在同一主机且属于同一进程，则支持 direct 模式
        if (ipeerInfo->hostHash == jpeerInfo->hostHash && ipeerInfo->pidHash == jpeerInfo->pidHash) directFlag = true;
        if (!supportFlag && directFlag) break;
      }
    }
  }
  *intraNodeP2pSupport = supportFlag;
  *directMode = directFlag;
  if (comm->rank == 0) INFO(NCCL_INIT, "Check P2P Type intraNodeP2pSupport %d directMode %d", supportFlag, directFlag);
  return ncclSuccess;
}
//建立p2p连接，ncclTransportP2pSetup 的 P2P 是广义上的两个设备之间的通信设置，包含 P2P、网络以及共享内存等。
ncclResult_t ncclTransportP2pSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, int connIndex) {
  // Stream used during transport setup; need for P2P pre-connect + CUDA Graph
  ncclResult_t ret = ncclSuccess;
  struct ncclConnect** data; // Store intermediate send/recvData structs for  存储临时的发送/接收连接结构体
  struct ncclConnect** recvData = NULL; // Points to entries inside data for given recv connection within a channel
  struct ncclConnect** sendData = NULL; // Points to entries inside data for given send connection within a channel
  int done = 0;
  int maxPeers = ncclParamConnectRoundMaxPeers(); // 获取最大并发 peers 数量，每次最多处理 maxPeers 个 peers

  struct timeval timeStart, timeLast;
  gettimeofday(&timeStart, NULL);
  timeLast = timeStart; // struct copy
  bool timeReported = false;

  NCCLCHECK(ncclCalloc(&data, maxPeers));//maxPeers个指针
  NCCLCHECKGOTO(ncclCalloc(&recvData, maxPeers), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&sendData, maxPeers), ret, fail);

  NCCLCHECKGOTO(ncclStrongStreamAcquireUncaptured(&comm->sharedRes->hostStream), ret, fail);
  // First time initialization
  for (int i=1; i<comm->nRanks; i++) {
    int bootstrapTag = (i<<8) + (graph ? graph->id+1 : 0);//// 标记用于唯一标识消息

    /*
    注意，对于ring连接来说，只有第一次循环中mask是有值的
    这里是遍历所有可能的peer。注意前面计算p2p schedule的时候，也是+delta和-delta，delta不超过nrank
    我没看懂recvPeer和sendPeer这样计算的逻辑...(我猜是因为这是个公共的函数，所以对与send和recv都要遍历所有的rank，这里把这两次遍历写在了一个循环里面)
    假设1 3.
    */
    int recvPeer = (comm->rank - i + comm->nRanks) % comm->nRanks;
    int sendPeer = (comm->rank + i) % comm->nRanks;
    uint64_t recvMask = comm->connectRecv[recvPeer];//如果这个peer在channel的掩码中没有记录，说明不需要
    uint64_t sendMask = comm->connectSend[sendPeer];

    // Data[i] contains all ncclConnect information for all send and receive connections with a given send and recv peer
    // This data is packed in the array based on the number of sendChannels and recvChannels connected with these peers
    // The first N entries contain recvData, connection information for recv connections
    // The next M entries contain sendData, connection information for send connections
    // It's not guaranteed that each entry of data has the same number of total or send/recv specific connections
    /*
        data[i] 包含了与特定peer的所有发送和接收连接的相关信息。
    这些信息根据与该对peer连接的发送通道（sendChannels）和接收通道（recvChannels）的数量进行组织。
    数组的前 N 个条目存储接收连接的信息（recvData），接下来的 M 个条目存储发送连接的信息（sendData）。
    每个 data 条目中的总连接数或特定的发送/接收连接数可能不同。
    */
    int p = i-(done+1);//p 表示当前 peer 在 data 数组中的索引，done是已经处理完的数量
    if (recvMask || sendMask) {
      if (data[p] == NULL) NCCLCHECKGOTO(ncclCalloc(data + p, 2 * MAXCHANNELS), ret, fail);
      else memset(data[p], 0, 2 * MAXCHANNELS * sizeof(struct ncclConnect));
    }
    recvData[p] = data[p];//
    int sendChannels = 0, recvChannels = 0;//相当于每个channel有一个ncclConnect
    int type;//传输方式
    TIME_START(0);
    for (int c=0; c<MAXCHANNELS; c++) {
      if (recvMask & (1UL<<c)) {
        //这里建立的连接是proxy的。
        NCCLCHECKGOTO(selectTransport<0>(comm, graph, recvData[p]+recvChannels++, c, recvPeer, connIndex, &type), ret, fail);
      }
    }
    TIME_STOP(0);
    TIME_START(1);
    sendData[p] = recvData[p]+recvChannels;
    for (int c=0; c<MAXCHANNELS; c++) {//如果设置了发送掩码，则选择对应的发送方式
      if (sendMask & (1UL<<c)) {
        NCCLCHECKGOTO(selectTransport<1>(comm, graph, sendData[p]+sendChannels++, c, sendPeer, connIndex, &type), ret, fail);
      }
    }
    TIME_STOP(1);

    TIME_START(2);
    if (sendPeer == recvPeer) {//当 发送目标和接收目标是同一个 rank 时，说明这是一个“自环”通信
      if (recvChannels+sendChannels) {
        //将 data[p] 中的内容发送给 recvPeer。这里的bootstrap是一次性连接，用于p2p的，与最开始的bootstrapAllgather是不同的。
        //让两个 rank 都知道对方的连接信息（如内存地址、协议类型等），才能建立真正的 P2P 通信链路
        NCCLCHECKGOTO(bootstrapSend(comm->bootstrap, recvPeer, bootstrapTag, data[p], sizeof(struct ncclConnect)*(recvChannels+sendChannels)), ret, fail);
        // recvPeer 接收它的连接信息到同样的 data[p] 缓冲区。
        NCCLCHECKGOTO(bootstrapRecv(comm->bootstrap, recvPeer, bootstrapTag, data[p], sizeof(struct ncclConnect)*(recvChannels+sendChannels)), ret, fail);
        sendData[p] = data[p];
        recvData[p] = data[p]+sendChannels;
      }
    } else {
      if (recvChannels) NCCLCHECKGOTO(bootstrapSend(comm->bootstrap, recvPeer, bootstrapTag, recvData[p], sizeof(struct ncclConnect)*recvChannels), ret, fail);
      if (sendChannels) NCCLCHECKGOTO(bootstrapSend(comm->bootstrap, sendPeer, bootstrapTag, sendData[p], sizeof(struct ncclConnect)*sendChannels), ret, fail);
      if (sendChannels) NCCLCHECKGOTO(bootstrapRecv(comm->bootstrap, sendPeer, bootstrapTag, sendData[p], sizeof(struct ncclConnect)*sendChannels), ret, fail);
      if (recvChannels) NCCLCHECKGOTO(bootstrapRecv(comm->bootstrap, recvPeer, bootstrapTag, recvData[p], sizeof(struct ncclConnect)*recvChannels), ret, fail);
    }
    TIME_STOP(2);
    //判断是否开始连接一批 peers，每次最多处理 maxPeers 个 peers。
    if (i-done == maxPeers || i == comm->nRanks-1) {
      // Loop until all channels with all ranks have been connected
      bool allChannelsConnected;
      allChannelsConnected = false;
      while (!allChannelsConnected) {
        allChannelsConnected = true;
        for (int j=done+1; j<=i; j++) {//处理done+1到i之间的所有peer
          //对于rank j，其recvPeer和sendPeer要计算一下
          int recvPeer = (comm->rank - j + comm->nRanks) % comm->nRanks;
          int sendPeer = (comm->rank + j) % comm->nRanks;
          uint64_t recvMask = comm->connectRecv[recvPeer];
          uint64_t sendMask = comm->connectSend[sendPeer];

          int p = j-(done+1);
          int sendDataOffset = 0;
          int recvDataOffset = 0;
          for (int c=0; c<MAXCHANNELS; c++) {
            TIME_START(3);
             // 建立发送连接
            if (sendMask & (1UL<<c)) {
              //这里前面的selectTransport已经设置好了conn的transportComm
              struct ncclConnector* conn = comm->channels[c].peers[sendPeer]->send + connIndex;
              // This connector hasn't completed connection yet 此时调用连接，尝试连接
              if (conn->connected == 0) {
                NCCLCHECKGOTO(conn->transportComm->connect(comm, sendData[p] + sendDataOffset, 1, comm->rank, conn), ret, fail);
                if (ret == ncclSuccess) {
                  conn->connected = 1;
                  /* 
                  comm->channels[c].devPeers[sendPeer]->send[connIndex] is a device memory access. 
                  将连接信息复制到 GPU 内存，供后续 kernel 使用；
                  */
                  CUDACHECKGOTO(cudaMemcpyAsync(&comm->channels[c].devPeersHostPtr[sendPeer]->send[connIndex], &conn->conn, sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice, comm->sharedRes->hostStream.cudaStream), ret, fail);
                } else if (ret == ncclInProgress) {
                  allChannelsConnected = false;
                }
              }
              sendDataOffset++;
            }
            TIME_STOP(3);

            // Start with recv channels
            TIME_START(4);
            if (recvMask & (1UL<<c)) { //同理，简历接收的连接
              struct ncclConnector* conn = comm->channels[c].peers[recvPeer]->recv + connIndex;
              // This connector hasn't completed connection yet
              if (conn->connected == 0) {
                NCCLCHECKGOTO(conn->transportComm->connect(comm, recvData[p] + recvDataOffset, 1, comm->rank, conn), ret, fail);
                if (ret == ncclSuccess) {
                  conn->connected = 1;
                  /* comm->channels[c].devPeers[recvPeer]->recv[connIndex] is a device memory access. */
                  CUDACHECKGOTO(cudaMemcpyAsync(&comm->channels[c].devPeersHostPtr[recvPeer]->recv[connIndex], &conn->conn, sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice, comm->sharedRes->hostStream.cudaStream), ret, fail);
                } else if (ret == ncclInProgress) {
                  allChannelsConnected = false;
                }
              }
              recvDataOffset++;
            }
            TIME_STOP(4);
          }
        }
        /*
          如果启用了 NCCL_REPORT_CONNECT_PROGRESS 参数，rank 0 会周期性打印连接进度；
          显示已完成百分比、耗时、预估剩余时间；
          用于调试和性能分析。
        */
        if (ncclParamReportConnectProgress() && comm->rank == 0 && done > 0) {
          struct timeval now;
          gettimeofday(&now, NULL);
          if (((now.tv_sec - timeLast.tv_sec) * 1.0 + (now.tv_usec - timeLast.tv_usec) * 1e-6) > 1) {
            float elapsed = (now.tv_sec - timeStart.tv_sec) * 1.0 + (now.tv_usec - timeStart.tv_usec) * 1e-6;
            float remaining = elapsed * (comm->nRanks - done) / done;
            printf("%sP2p connect: %g%% Elapsed %d:%02d Remaining %d:%02d                                       ",
              timeReported ? "\r" : "", done * 100.0 / comm->nRanks, ((int)elapsed) / 60, ((int)elapsed) % 60, ((int)remaining) / 60, ((int)remaining) % 60);
            fflush(stdout);
            timeReported = true;
            timeLast = now; // struct copy;
          }
        }
      }
      done = i;
    }
  }

  {
    struct timeval now;
    gettimeofday(&now, NULL);
    float elapsed = (now.tv_sec - timeStart.tv_sec)*1.0 + (now.tv_usec-timeStart.tv_usec)*1e-6;
    if (elapsed > 1.0) INFO(NCCL_PROFILE, "timings: rank %d nranks %d P2p connect done in %.2f", comm->rank, comm->nRanks, elapsed);
    if (timeReported) {
      printf("\rP2p connect done in %d:%02d                                                                       \n",
             ((int)elapsed)/60, ((int)elapsed)%60);
      fflush(stdout);
    }
  }

  /* We need to sync ranks here since some ranks might run too fast after connection setup
   * and start to destroy the connection after returning from this function; however, the
   * others might still be trying to connect and import the buffer. No sync can lead to invalid
   * shmem/cuda buffer. In addition, we also clear all connect masks and free each connectInfo array 
   * 
   * 同步所有rank的连接，防止某些 rank 提前退出导致其他 rank 访问无效内存；确保所有 rank 都完成了连接初始化；
   * 所以 NCCL 在这里使用了 双向握手（send + recv） 来确保：
   * */
  for (int i = 1; i < comm->nRanks; i++) {
    int bootstrapTag = (i << 8) + (1 << 7) + (graph ? graph->id + 1 : 0);
    int recvPeer = (comm->rank - i + comm->nRanks) % comm->nRanks;
    int sendPeer = (comm->rank + i) % comm->nRanks;

    if (recvPeer != sendPeer) {
      //只要存在channel就行，说明对这个peer有连接。发送一个空消息（size=0）表示“我准备好了”；然后接收对方的空消息，表示“你也准备好了”；
      if (comm->connectSend[sendPeer] != 0UL) NCCLCHECKGOTO(bootstrapSend(comm->bootstrap, sendPeer, bootstrapTag, NULL, 0), ret, fail);
      if (comm->connectRecv[recvPeer] != 0UL) NCCLCHECKGOTO(bootstrapSend(comm->bootstrap, recvPeer, bootstrapTag, NULL, 0), ret, fail);
      if (comm->connectSend[sendPeer] != 0UL) NCCLCHECKGOTO(bootstrapRecv(comm->bootstrap, sendPeer, bootstrapTag, NULL, 0), ret, fail);
      if (comm->connectRecv[recvPeer] != 0UL) NCCLCHECKGOTO(bootstrapRecv(comm->bootstrap, recvPeer, bootstrapTag, NULL, 0), ret, fail);
    } else {
      if (comm->connectSend[sendPeer] != 0UL || comm->connectRecv[recvPeer] != 0UL) {
        NCCLCHECKGOTO(bootstrapSend(comm->bootstrap, sendPeer, bootstrapTag, NULL, 0), ret, fail);
        NCCLCHECKGOTO(bootstrapRecv(comm->bootstrap, sendPeer, bootstrapTag, NULL, 0), ret, fail);
      }
    }
    comm->connectRecv[recvPeer] = comm->connectSend[sendPeer] = 0UL;//清除 connectRecv 和 connectSend 掩码,表示与这个 peer 的连接已经完成；

  }

  TIME_PRINT("P2P Setup/Connect");
exit:
  for(int i=0; i<maxPeers; ++i){
    if(data[i]) free(data[i]);
  }
  free(data);
  if (sendData) free(sendData);
  if (recvData) free(recvData);

  NCCLCHECK(ncclStrongStreamWaitStream(ncclCudaGraphNone(), &comm->sharedRes->deviceStream, &comm->sharedRes->hostStream));
  NCCLCHECK(ncclStrongStreamRelease(ncclCudaGraphNone(), &comm->sharedRes->hostStream));
  return ret;
fail:
  goto exit;
}

extern struct ncclTransport collNetTransport;

// All ranks must participate in collNetSetup call
// We do not NCCLCHECK this call because we would fall back to P2P network in case CollNet setup fails
bool ncclTransportCollNetSetup(struct ncclComm* comm, struct ncclTopoGraph* collNetGraph, struct ncclChannel* channel, int masterRank, int masterPeer, int collNetGraphChannelId, int type, ncclConnect* connect) {
  ncclResult_t ret = ncclSuccess;
  int rank = comm->rank;
  int nranks = comm->nRanks;
  int nMasters = comm->nNodes;
  int isMaster = (rank == masterRank) ? 1 : 0;

  // check if we can connect to collnet, whose root is the nranks-th rank
  struct ncclPeerInfo *myInfo = comm->peerInfo+rank, *peerInfo = comm->peerInfo+nranks;
  peerInfo->rank = nranks;

  if (isMaster && type == collNetSend) {
    TRACE(NCCL_INIT, "CollNet [send] : rank %d collNetRank %d collNetNranks %d received connect from rank %d", rank, comm->node, nMasters, masterPeer);
  }

  // select
  struct ncclChannelPeer* root = channel->peers[nranks];
  // connector index: 0 for recv, 1 for send
  struct ncclConnector* conn = (type == collNetRecv) ? root->recv+type : root->send+type;
  struct ncclTransportComm* transportComm = (type == collNetRecv) ? &(collNetTransport.recv) : &(collNetTransport.send);
  conn->transportComm = transportComm;
  // setup
  struct ncclConnect myConnect = { 0 };
  struct {
    int isMaster;
    ncclConnect connect;
  } *allConnects = NULL;
  ncclConnect *masterConnects = NULL;
  if (isMaster) {
    NCCLCHECK(transportComm->setup(comm, collNetGraph, myInfo, peerInfo, &myConnect, conn, collNetGraphChannelId, type));
  }
  // prepare connect handles
  NCCLCHECK(ncclCalloc(&masterConnects, nMasters));
  if (type == collNetRecv) {  // recv side: AllGather
    // all ranks must participate
    NCCLCHECKGOTO(ncclCalloc(&allConnects, nranks), ret, cleanup);
    allConnects[rank].isMaster = isMaster;
    memcpy(&(allConnects[rank].connect), &myConnect, sizeof(struct ncclConnect));
    NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, allConnects, sizeof(*allConnects)), ret, cleanup);
    // consolidate
    int c = 0;
    for (int r = 0; r < nranks; r++) {
      if (allConnects[r].isMaster) {
        memcpy(masterConnects+c, &(allConnects[r].connect), sizeof(struct ncclConnect));
        c++;
      }
    }
  } else { // send side : copy in connect info received from peer recv master
    if (isMaster) memcpy(masterConnects+comm->node, connect, sizeof(struct ncclConnect));
  }
  // connect
  if (isMaster) {
    NCCLCHECKGOTO(transportComm->connect(comm, masterConnects, nMasters, comm->node, conn), ret, cleanup);
    struct ncclDevChannelPeer* devRoot;
    CUDACHECKGOTO(cudaMemcpy(&devRoot, channel->devPeers + nranks, sizeof(struct ncclDevChannelPeer*), cudaMemcpyDeviceToHost), ret, cleanup);
    struct ncclConnInfo* devConnInfo = (type == collNetRecv) ? devRoot->recv + type : devRoot->send + type;
    CUDACHECKGOTO(cudaMemcpy(devConnInfo, &conn->conn, sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice), ret, cleanup);
  }
  if (isMaster && type == collNetRecv) {
    memcpy(connect, masterConnects+comm->node, sizeof(struct ncclConnect));
    TRACE(NCCL_INIT, "CollNet [recv] : rank %d collNetRank %d collNetNranks %d sent connect to rank %d", rank, comm->node, nMasters, masterPeer);
  }
cleanup:
  if (allConnects != NULL) free(allConnects);
  if (masterConnects != NULL) free(masterConnects);
  return ret != ncclSuccess;
}

ncclResult_t ncclTransportCollNetCheck(struct ncclComm* comm, int collNetSetupFail) {
  // AllGather collNet setup results
  int allGatherFailures[NCCL_MAX_LOCAL_RANKS] = {0};
  allGatherFailures[comm->localRank] = collNetSetupFail;
  NCCLCHECK(bootstrapIntraNodeAllGather(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks, allGatherFailures, sizeof(int)));
  for (int i=0; i<comm->localRanks; i++) {
    if (allGatherFailures[i] != 0) {
      collNetSetupFail = 1;
      break;
    }
  }
  if (collNetSetupFail) {
    if (comm->localRank == 0) WARN("Cannot initialize CollNet, using point-to-point network instead");
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t ncclTransportCollNetFree(struct ncclComm* comm) {
  // Free collNet resources
  for (int r=0; r<comm->nChannels; r++) {
    struct ncclChannel* channel = comm->channels+r;
    struct ncclChannelPeer* peer = channel->peers[comm->nRanks];
    if (peer) {
      if (ncclAtomicRefCountDecrement(&peer->refCount) == 0) {
        for (int b=0; b<NCCL_MAX_CONNS; b++) {
          struct ncclConnector* send = peer->send + b;
          if (send->transportResources && send->transportComm) NCCLCHECK(send->transportComm->free(send));
          send->transportResources = NULL; // avoid double free
        }
        for (int b=0; b<NCCL_MAX_CONNS; b++) {
          struct ncclConnector* recv = peer->recv + b;
          if (recv->transportResources && recv->transportComm) NCCLCHECK(recv->transportComm->free(recv));
          recv->transportResources = NULL; // avoid double free
        }
      }
    }
  }
  return ncclSuccess;
}
