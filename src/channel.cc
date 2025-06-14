/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "channel.h"
#include "param.h"
#include "gdrwrap.h"
#include "transport.h"
//用于初始化 NCCL 通信通道（channel）的各种资源，包括主机端和设备端的 peer 结构、ring 结构等，为后续的通信操作做准备。
//主要就是分配peers、devPeers内存，然后和共享资源绑定，这个过程会涉及到cuda graph、stream等
//让给定channel的devPeers指针与全局共享的那个是一样的
ncclResult_t initChannel(struct ncclComm* comm, int channelId) {
  struct ncclChannel* channel = &comm->channels[channelId];
  if (channel->id != -1) return ncclSuccess;

  int nRanks = comm->nRanks;
  int nvlsRanks = comm->localRanks;
  //计算 peer 数量 nPeers = nRanks + 1 + nvlsRanks ，其中 +1 是为 Collnet root（网络通信）， +nvlsRanks 是为 NVLS（NVIDIA Local Switch）通信。
  int nPeers = nRanks + 1 /* Collnet */ + nvlsRanks /* NVLS */;
  channel->id = channelId;
  channel->workFifoProduced = 0;

  struct ncclSharedResources* sharedRes = comm->sharedRes;
  //获取并锁定 device stream，调用 ncclStrongStreamAcquireUncaptured 获取设备流，保证后续异步 CUDA 操作的正确性。
  NCCLCHECK(ncclStrongStreamAcquireUncaptured(&sharedRes->deviceStream));
  //初始化主机端 peers
  if (channel->peers == NULL) {
    // The extra on nRanks+1 is for collnet root (i.e. network)
    // Allocate everything related to sharedRes with ncclCalloc as this can be
    // shared between communicators hence should not be tied to comm.
    if (sharedRes->peers[channelId] == NULL) {
      NCCLCHECK(ncclCalloc(sharedRes->peers + channelId, sharedRes->tpNRanks));
    }
    channel->peers = ncclMemoryStackAlloc<struct ncclChannelPeer*>(&comm->memPermanent, nPeers);
    for (int r = 0; r < nRanks; r++) {
      //将 channel->peers[r] 指向全局共享的 peer，并增加引用计数。
      channel->peers[r] = comm->sharedRes->peers[channelId] + comm->topParentRanks[r];
      ncclAtomicRefCountIncrement(&channel->peers[r]->refCount);
    }
  }
  //初始化设备端 peers
  if (channel->devPeers == NULL) {
    if (sharedRes->devPeers[channelId] == NULL) {
      //若 sharedRes->devPeers[channelId] 为空，则为其分配设备端内存。
      NCCLCHECK(ncclCudaCallocAsync(sharedRes->devPeers + channelId, sharedRes->tpNRanks, sharedRes->deviceStream.cudaStream));
    }
    /* channel->devPeers is not shared, so just free it when calling commFree() */
    //为本通道分配设备端 peers 内存，并注册释放回调。
    NCCLCHECK(ncclCudaCallocAsync(&channel->devPeers, nPeers, sharedRes->deviceStream.cudaStream));
    ncclCommPushCudaFree(comm, channel->devPeers);
    //分配主机端指针数组 devPeersHostPtr
    NCCLCHECK(ncclCalloc(&channel->devPeersHostPtr, nPeers));
    for (int r = 0; r < nRanks; r++) {//遍历所有 rank，将全局共享的设备端 peer 地址拷贝到本通道的 devPeers ，并同步到主机端指针。
      uintptr_t addr = (uintptr_t)(comm->sharedRes->devPeers[channelId] + comm->topParentRanks[r]);
      NCCLCHECK(ncclCudaMemcpyAsync((uintptr_t*)(channel->devPeers + r), (uintptr_t*)&addr, 1, sharedRes->deviceStream.cudaStream));
      channel->devPeersHostPtr[r] = (struct ncclDevChannelPeer*)addr;
    }
  }
  //为 ring 的 userRanks 分配内存。
  channel->ring.userRanks = ncclMemoryStackAlloc<int>(&comm->memPermanent, nRanks);
  //为设备端 ring ranks 分配内存，并注册释放回调。
  NCCLCHECK(ncclCudaCallocAsync(&channel->devRingUserRanks, nRanks, sharedRes->deviceStream.cudaStream));
  ncclCommPushCudaFree(comm, channel->devRingUserRanks);

  /* guarantee addr has been copied into channel->devPeers */
  NCCLCHECK(ncclStrongStreamSynchronize(&sharedRes->deviceStream));//保证所有异步拷贝完成后同步设备流。
  NCCLCHECK(ncclStrongStreamRelease(ncclCudaGraphNone(), &sharedRes->deviceStream));//如果需要，在 stream 上记录一个事件，为后续可能的 Graph 捕获或 NCCL 操作提供一个同步点。

  return ncclSuccess;
}

ncclResult_t initNvlsChannel(struct ncclComm* comm, int channelId, struct ncclComm* parent, bool share) {
  struct ncclChannel* channel = &comm->channels[channelId];
  struct ncclSharedResources* sharedRes = comm->sharedRes;

  if (channel->nvlsPeers != NULL)
    return ncclSuccess;

  if (channel->id == -1)
    NCCLCHECK(initChannel(comm, channelId));

  NCCLCHECK(ncclStrongStreamAcquireUncaptured(&sharedRes->deviceStream));

  int nvlsRanks = comm->localRanks;

  if (share) {
    channel->nvlsPeers = parent->channels[channelId].nvlsPeers;
    channel->nvlsDevPeers = parent->channels[channelId].nvlsDevPeers;
    for (int r = 0; r < nvlsRanks; ++r) {
      int tr = comm->topParentLocalRanks[r];
      uintptr_t addr = (uintptr_t)(parent->channels[channelId].nvlsDevPeers + tr);
      channel->peers[comm->nRanks + 1 + r] = parent->channels[channelId].nvlsPeers + tr;
      NCCLCHECK(ncclCudaMemcpyAsync((uintptr_t*)(channel->devPeers + comm->nRanks + 1 + r), (uintptr_t*)&addr, 1, sharedRes->deviceStream.cudaStream));
      channel->devPeersHostPtr[comm->nRanks + 1 + r] = (struct ncclDevChannelPeer*)addr;
      ncclAtomicRefCountIncrement(&parent->channels[channelId].nvlsPeers[tr].refCount);
    }
  } else {
    NCCLCHECK(ncclCalloc(&channel->nvlsPeers, nvlsRanks));
    NCCLCHECK(ncclCudaCallocAsync(&channel->nvlsDevPeers, nvlsRanks, sharedRes->deviceStream.cudaStream));
    for (int r = 0; r < nvlsRanks; ++r) {
      uintptr_t addr = (uintptr_t)(channel->nvlsDevPeers + r);
      channel->peers[comm->nRanks + 1 + r] = channel->nvlsPeers + r;
      NCCLCHECK(ncclCudaMemcpyAsync((uintptr_t*)(channel->devPeers + comm->nRanks + 1 + r), (uintptr_t*)&addr, 1, sharedRes->deviceStream.cudaStream));
      channel->devPeersHostPtr[comm->nRanks + 1 + r] = (struct ncclDevChannelPeer*)addr;
      ncclAtomicRefCountIncrement(&channel->nvlsPeers[r].refCount);
    }
  }

  NCCLCHECK(ncclStrongStreamSynchronize(&sharedRes->deviceStream));
  NCCLCHECK(ncclStrongStreamRelease(ncclCudaGraphNone(), &sharedRes->deviceStream));

  return ncclSuccess;
}

ncclResult_t initCollnetChannel(struct ncclComm* comm, int channelId, struct ncclComm* parent, bool share) {
  struct ncclChannel* channel = &comm->channels[channelId];
  struct ncclSharedResources* sharedRes = comm->sharedRes;
  uintptr_t addr;

  if (channel->collnetPeers != NULL)
    return ncclSuccess;

  if (channel->id == -1)
    NCCLCHECK(initChannel(comm, channelId));

  NCCLCHECK(ncclStrongStreamAcquireUncaptured(&sharedRes->deviceStream));

  if (share) {
    channel->collnetPeers = parent->channels[channelId].collnetPeers;
    channel->collnetDevPeers = parent->channels[channelId].collnetDevPeers;
    addr = (uintptr_t)parent->channels[channelId].collnetDevPeers;
    channel->peers[comm->nRanks] = parent->channels[channelId].collnetPeers;
    NCCLCHECK(ncclCudaMemcpyAsync((uintptr_t*)(channel->devPeers + comm->nRanks), (uintptr_t*)&addr, 1, sharedRes->deviceStream.cudaStream));
    channel->devPeersHostPtr[comm->nRanks] = (struct ncclDevChannelPeer*)addr;
    ncclAtomicRefCountIncrement(&parent->channels[channelId].collnetPeers->refCount);
  } else {
    NCCLCHECK(ncclCalloc(&channel->collnetPeers, 1));
    NCCLCHECK(ncclCudaCallocAsync(&channel->collnetDevPeers, 1, sharedRes->deviceStream.cudaStream));
    addr = (uintptr_t)channel->collnetDevPeers;
    channel->peers[comm->nRanks] = channel->collnetPeers;
    NCCLCHECK(ncclCudaMemcpyAsync((uintptr_t*)(channel->devPeers + comm->nRanks), (uintptr_t*)&addr, 1, sharedRes->deviceStream.cudaStream));
    channel->devPeersHostPtr[comm->nRanks] = (struct ncclDevChannelPeer*)addr;
    ncclAtomicRefCountIncrement(&channel->collnetPeers->refCount);
  }

  NCCLCHECK(ncclStrongStreamSynchronize(&sharedRes->deviceStream));
  NCCLCHECK(ncclStrongStreamRelease(ncclCudaGraphNone(), &sharedRes->deviceStream));

  return ncclSuccess;
}

ncclResult_t freeChannel(struct ncclChannel* channel, int nRanks, int collnetNRanks, int nvlsNRanks) {
  int nPeers = nRanks + collnetNRanks + nvlsNRanks;
  /* channel peers are only valid when async init thread completes commAlloc() and
   * the channel is intialized with initChannel(); if either is not done, this channel
   * should never be free. */
  if (channel->id == -1 || channel->peers == NULL) return ncclSuccess;

  // Free transport proxy resources
  // Note: free all send resources first due to CollNet arrangement
  for (int r = 0; r < nPeers; r++) {
    struct ncclChannelPeer* peer = channel->peers[r];
    if (peer) {
      if (ncclAtomicRefCountDecrement(&peer->refCount) == 0) {
        for (int b=0; b<NCCL_MAX_CONNS; b++) {
          if (peer->send[b].transportComm) NCCLCHECK(peer->send[b].transportComm->free(peer->send+b));
          if (peer->recv[b].transportComm) NCCLCHECK(peer->recv[b].transportComm->free(peer->recv+b));
        }
        if (r == nRanks) {
          free(channel->collnetPeers);
          ncclCudaFree(channel->collnetDevPeers);
        } else if (r == nPeers - 1) {
          free(channel->nvlsPeers);
          ncclCudaFree(channel->nvlsDevPeers);
        }
      }
    }
  }

  free(channel->devPeersHostPtr);
  return ncclSuccess;
}
