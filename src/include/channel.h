/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_CHANNEL_H_
#define NCCL_CHANNEL_H_
#include "comm.h"
#include "utils.h"

#include <algorithm>

ncclResult_t initChannel(struct ncclComm* comm, int channelid);
ncclResult_t initNvlsChannel(struct ncclComm* comm, int channelId, struct ncclComm* parent, bool share);
ncclResult_t initCollnetChannel(struct ncclComm* comm, int channelId, struct ncclComm* parent, bool share);
ncclResult_t freeChannel(struct ncclChannel* channel, int nRanks, int collnetNRanks, int nvlsNRanks);
//根据comm和p2pRound计算出当前的channel id，这里的计算我理解的是相当于给每个channel id分配了batch的rank。
inline uint8_t ncclP2pChannelBaseForRound(struct ncclComm* comm, int p2pRound) {
  if (comm->nNodes > 1) {
    int nodeDelta = p2pRound/comm->maxLocalRanks;//因为round肯定是优先按照本地rank排列的，并且会分配完本地所有rank，所以可以根据这个算出当前是第几个node
    int localDelta = p2pRound%comm->maxLocalRanks;
    int base = nodeDelta*divUp(comm->maxLocalRanks, NCCL_MAX_DEV_WORK_P2P_PER_BATCH);
    base += localDelta/NCCL_MAX_DEV_WORK_P2P_PER_BATCH;
    return base & 0xff;
  } else {
    return p2pRound & 0xff;
  }
}

#endif
