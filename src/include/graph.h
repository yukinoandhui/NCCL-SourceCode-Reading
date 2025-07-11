/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_GRAPH_H_
#define NCCL_GRAPH_H_

#include "nccl.h"
#include "device.h"
#include <limits.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdio.h>
#include <sched.h>

ncclResult_t ncclTopoCudaPath(int cudaDev, char** path);

struct ncclTopoSystem;
// Build the topology
ncclResult_t ncclTopoGetSystem(struct ncclComm* comm, struct ncclTopoSystem** system, const char* dumpXmlFile=NULL);
ncclResult_t ncclTopoSortSystem(struct ncclTopoSystem* system);
ncclResult_t ncclTopoPrint(struct ncclTopoSystem* system);

ncclResult_t ncclTopoComputePaths(struct ncclTopoSystem* system, struct ncclComm* comm);
void ncclTopoFree(struct ncclTopoSystem* system);
ncclResult_t ncclTopoTrimSystem(struct ncclTopoSystem* system, struct ncclComm* comm);
ncclResult_t ncclTopoComputeP2pChannels(struct ncclComm* comm);
ncclResult_t ncclTopoGetNvbGpus(struct ncclTopoSystem* system, int rank, int* nranks, int** ranks);
ncclResult_t ncclTopoPathAllNVLink(struct ncclTopoSystem* system, int* allNvLink);

ncclResult_t ncclTopoComputeCommCPU(struct ncclComm* comm);

// Query topology
ncclResult_t ncclTopoGetNetDev(struct ncclComm* comm, int rank, struct ncclTopoGraph* graph, int channelId, int peerRank, int64_t* id, int* dev, int* proxyRank);
ncclResult_t ncclTopoCheckP2p(struct ncclComm* comm, struct ncclTopoSystem* system, int rank1, int rank2, int* p2p, int *read, int* intermediateRank);
ncclResult_t ncclTopoCheckMNNVL(struct ncclTopoSystem* system, struct ncclPeerInfo* info1, struct ncclPeerInfo* info2, int* ret);
ncclResult_t ncclTopoCheckGdr(struct ncclTopoSystem* topo, int rank, int64_t netId, int read, int* useGdr);
ncclResult_t ncclTopoNeedFlush(struct ncclComm* comm, int netDev, int rank, int* flush);
ncclResult_t ncclTopoIsGdrAvail(struct ncclTopoSystem* system, int rank, bool *avail);
ncclResult_t ncclTopoCheckNet(struct ncclTopoSystem* system, int rank1, int rank2, int* net);
int ncclPxnDisable(struct ncclComm* comm);
ncclResult_t ncclTopoGetPxnRanks(struct ncclComm* comm, int** intermediateRanks, int* nranks);
ncclResult_t ncclGetLocalCpu(struct ncclTopoSystem* system, int gpu, int* retCpu);

// Find CPU affinity
ncclResult_t ncclTopoGetCpuAffinity(struct ncclTopoSystem* system, int rank, cpu_set_t* affinity);

#define NCCL_TOPO_CPU_ARCH_X86 1
#define NCCL_TOPO_CPU_ARCH_POWER 2
#define NCCL_TOPO_CPU_ARCH_ARM 3
#define NCCL_TOPO_CPU_ARCH_MIXED 4
#define NCCL_TOPO_CPU_VENDOR_INTEL 1
#define NCCL_TOPO_CPU_VENDOR_AMD 2
#define NCCL_TOPO_CPU_VENDOR_ZHAOXIN 3
#define NCCL_TOPO_CPU_VENDOR_MIXED 4
#define NCCL_TOPO_CPU_TYPE_BDW 1
#define NCCL_TOPO_CPU_TYPE_SKL 2
#define NCCL_TOPO_CPU_TYPE_YONGFENG 1
ncclResult_t ncclTopoCpuType(struct ncclTopoSystem* system, int* arch, int* vendor, int* model);
ncclResult_t ncclTopoGetGpuCount(struct ncclTopoSystem* system, int* count);
ncclResult_t ncclTopoGetNetCount(struct ncclTopoSystem* system, int* count);
ncclResult_t ncclTopoGetNvsCount(struct ncclTopoSystem* system, int* count);
ncclResult_t ncclTopoGetLocalNet(struct ncclTopoSystem* system, int rank, int channelId, int64_t* id, int* dev);
ncclResult_t ncclTopoGetLocalGpu(struct ncclTopoSystem* system, int64_t netId, int* gpuIndex);
ncclResult_t getLocalNetCountByBw(struct ncclTopoSystem* system, int gpu, int *count);

#define NCCL_TOPO_MAX_NODES 256 //- 这个值是 单一节点（单机）内的最大节点数 ，不是整个集群的总 GPU 数。 这里的“节点”不仅仅指 GPU，还包括 CPU、PCI、NIC、NVS 等所有类型的拓扑节点。

// Init search. Needs to be done before calling ncclTopoCompute
ncclResult_t ncclTopoSearchInit(struct ncclTopoSystem* system);
//平衡树(Balanced Tree) - 在两个GPU之间分配NIC流量 平衡树模式下，所有 GPU 以尽量平衡的二叉树结构连接，父节点和子节点可以分布在不同的主机上。树的根节点通常在第一台机器，子节点均匀分布在所有机器上，目的是让数据在树上传递时，跨节点流量尽量均衡。
#define NCCL_TOPO_PATTERN_BALANCED_TREE 1   // Spread NIC traffic between two GPUs (Tree parent + one child on first GPU, second child on second GPU)
//分裂树(Split Tree) - 第一个GPU作为树的父节点，第二个GPU作为子节点
 //分裂树模式下，树的父节点和子节点 有意分布在不同的主机上 ，比如根节点在一台机器，所有子节点在另一台机器。这样可以让跨节点流量集中在根节点与其它节点之间，适合某些特殊的网络结构或优化场景。
#define NCCL_TOPO_PATTERN_SPLIT_TREE 2      // Spread NIC traffic between two GPUs (Tree parent on first GPU, tree children on the second GPU)
//树(Tree) - 所有NIC流量都通过同一个GPU
#define NCCL_TOPO_PATTERN_TREE 3            // All NIC traffic going to/from the same GPU
//环(Ring) - 环形通信模式
#define NCCL_TOPO_PATTERN_RING 4            // Ring
//NVLS+SHARP和NVLS+Tree模式。 在节点内部使用NVLink高带宽连接，在节点间SHARP通过在网络交换机（如 InfiniBand 交换机）上实现数据聚合和还原操作。NVLS tree本质上是机内采用NVLS，机间采用tree
#define NCCL_TOPO_PATTERN_NVLS 5            // // NVLS+SHARP and NVLS+Tree
//直接CollNet模式，collNet是NCCL中的一个插件，用户可自定义通信方式。例如自定义在网计算，减少CPU和PCIe开销。 NCCL-SHARP就是一个插件案例，允许NCCL 利用支持 SHARP 的网络设备进行高效的集体通信。
//SHARP 是 Mellanox 开发的一种在网络交换机层面实现数据聚合和规约的协议,NCCL 通过 CollNet 插件接口加载 SHARP 插件
#define NCCL_TOPO_PATTERN_COLLNET_DIRECT 6  // Collnet Direct
//用于建模和记录 NCCL 在不同通信算法（如 Ring、Tree、CollNet、NVLS 等）下，节点间和节点内的通信路径、带宽、通道等信息
//主要是记录了搜索的channel的结果。
struct ncclTopoGraph {
  // Input / output
  int id; // ring : 0, tree : 1, collnet : 2, nvls : 3, collnetDirect : 4  。表示拓扑图的类型，0: 环形(Ring)算法，3: NVLS算法
  int pattern;// 通信模式，定义了数据如何在GPU和网络接口之间流动
  int crossNic;//是否跨网卡通信 跨网卡通信（crossNic） ，就是指在单节点内部，合理分配GPU到不同NIC的通信路径，使得多块网卡都能被充分利用。
  int collNet;//是否使用CollNet网络
  int minChannels;//最小通道数
  int maxChannels;
  // Output
  int nChannels;//实际使用的通道数
  float bwIntra;//节点内单个channel的带宽(GB/s)
  float bwInter;//节点间单个channel的带宽(GB/s)
  float latencyInter;//节点间延迟(μs)
  int typeIntra;//节点内连接类型(如NVLink、PCIe等)
  int typeInter;//节点间连接类型(如InfiniBand、以太网等)
  int sameChannels;//是否所有通道使用相同的路径(硬件路径)
  int nHops;//通信路径中的跳数
  //channelId* system->nodes[GPU].count就是每个通道的头部gpu的rank。它是一个一维数组，但可以理解为二维结构： intra[channelId * gpuCount + gpuIndex] 。
  int intra[MAXCHANNELS*NCCL_TOPO_MAX_NODES];//节点内通信路径数组,存储每个通道channel内部的通信路径，用于指导节点内GPU之间的数据传输。也就是存的gpu rank
  int64_t inter[MAXCHANNELS*2];//是一个记录每个通道所选用的NET（NIC）id的数组。(一般是两个，我的理解是，跨网卡的话，ring模式下会出现这种情况：net1->gpu0->gpu1->net2)
};
ncclResult_t ncclTopoCompute(struct ncclTopoSystem* system, struct ncclTopoGraph* graph);

ncclResult_t ncclTopoPrintGraph(struct ncclTopoSystem* system, struct ncclTopoGraph* graph);
ncclResult_t ncclTopoDumpGraphs(struct ncclTopoSystem* system, int ngraphs, struct ncclTopoGraph** graphs);
//存储不同通信算法下每个通道的进程关系
struct ncclTopoRanks {
  int ringRecv[MAXCHANNELS];// ringRecv[c]：当前rank所在的”部分 channel c ”的头节点，也可以理解为所在机器的头节点rank
  int ringSend[MAXCHANNELS];// 当前rank所在的”部分 channel c ”的尾节点也可以理解为所在机器的尾节点rank；
  int ringPrev[MAXCHANNELS];// 每个通道中当前rank在环中的前一个rank
  int ringNext[MAXCHANNELS];// 每个通道中当前rank在环中的下一个rank
  int treeToParent[MAXCHANNELS];// 每个通道中当前rank在树形通信中的父节点rank。 不过我的理解是这个node的中的头rank
  int treeToChild0[MAXCHANNELS];// 每个通道中当前rank在树形通信中的第一个子节点rank
  int treeToChild1[MAXCHANNELS]; // 每个通道中当前rank在树形通信中的第二个子节点rank
  int nvlsHeads[MAXCHANNELS];// NVLS算法中，每个通道的head节点rank
  int nvlsHeadNum; // NVLS算法中head节点的数量  NVLink SHARP
  /*NVLS算法利用NVLink带宽在节点内高效聚合/分发数据，然后通过少量head节点与外部网络通信，实现节点间高效数据交换 。
  节点内所有GPU通过NVLink高速互联，形成一个高带宽的“全互连”或“近似全互连”结构。每个节点只选取少数几个GPU作为“head节点”（通常是与NIC直连或带宽最优的GPU）
  只有head节点负责与其他节点的head节点进行跨节点通信（通常通过InfiniBand/Ethernet等网络）。
  其他非head GPU的数据需要先通过NVLink传递到本节点的head节点，再由head节点发送到外部。
  */
};

ncclResult_t ncclTopoPreset(struct ncclComm* comm, struct ncclTopoGraph** graphs, struct ncclTopoRanks* topoRanks);

ncclResult_t ncclTopoPostset(struct ncclComm* comm, int* firstRanks, int* treePatterns,
    struct ncclTopoRanks** allTopoRanks, int* rings, struct ncclTopoGraph** graphs, struct ncclComm* parent);

ncclResult_t ncclTopoTuneModel(struct ncclComm* comm, int minCompCap, int maxCompCap, struct ncclTopoGraph** graphs);
ncclResult_t ncclTopoGetAlgoTime(struct ncclComm* comm, int coll, int algorithm, int protocol, size_t nBytes, int numPipeOps, float* time);

#endif
