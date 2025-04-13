/*************************************************************************
 * Copyright (c) 2016-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_RAS_H_
#define NCCL_RAS_H_

#include "socket.h"

// Structure used to communicate data about NCCL ranks from NCCL threads to RAS.
//用于监控和管理分布式训练中的GPU集群健康状态
// reliability, availability, and serviceability.可靠性、可用性和可服务性
struct rasRankInit {
  union ncclSocketAddress addr;
  pid_t pid;
  int cudaDev;
  int nvmlDev;//NVML设备ID，用于通过NVIDIA管理库监控GPU状态。NVIDIA Management Library (NVML)
};

ncclResult_t ncclRasCommInit(struct ncclComm* comm, struct rasRankInit* myRank);
ncclResult_t ncclRasCommFini(const struct ncclComm* comm);
ncclResult_t ncclRasAddRanks(struct rasRankInit* ranks, int nranks);

#endif // !NCCL_RAS_H_
