/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_TOPO_H_
#define NCCL_TOPO_H_

#include "graph.h"
#include "core.h"
//GB/s
#define LOC_BW 5000.0 // Local bandwidth  本地带宽（GB/s），用于本地通信建模。 这里的值估计都不是准确的，大小只是表示一种相对顺序，本地的最快
#define SM60_NVLINK_BW 18.0
#define SM70_NVLINK_BW 20.0
#define SM80_NVLINK_BW 20.0
#define SM90_NVLINK_BW 20.6
#define SM86_NVLINK_BW 12.0
#define SM100_NVLINK_BW 40.0
#define PCI_BW 12.0           // PCI Gen3 x16
#define QPI_BW 6.0 //QPI（Intel CPU互连）带宽
#define AMD_BW 16.0//AMD CPU互连带宽
#define SKL_QPI_BW 10.0 //Skylake QPI带宽
#define ZPI_BW 6.0 //ZPI带宽
#define YONGFENG_ZPI_BW 9.0 //Yongfeng ZPI带宽
#define P9_BW 32.0//Power9 CPU互连带宽
#define ARM_BW 6.0//ARM CPU互连带宽
#define NET_BW 12.0           // 100Gbit 网络带宽

// Intel CPU convert GPU P2P traffic into 64B PCI TLPs, so GPU
// to GPU traffic consumes more PCI bandwidth. Intel CPU下GPU直连通信的PCIe带宽开销修正（多20%）
#define INTEL_P2P_OVERHEAD(bw) (bw*6/5)

#define NCCL_TOPO_NODE_TYPES 6 //拓扑节点类型总数为6。
#define GPU 0//GPU节点类型编号。
#define PCI 1//PCI节点类型编号
#define NVS 2//NVSwitch节点类型编号
#define CPU 3 // Actually NUMA domains CPU节点类型编号（实际为NUMA域）。
#define NIC 4 //网络接口卡（NIC）节点类型编号。
#define NET 5 //网络节点类型编号。
extern const char* topoNodeTypeStr[];//网络节点类型编号

// We want link types and path types to match as much as possible
#define LINK_LOC 0 //跟自己相连的边
#define LINK_NVL 1 //NVLink 连接的边
// Skipping 2 for PATH_NVB
#define LINK_PCI 3 //PCIe总线连接的
// Skipping 4 for PATH_PXB
// Skipping 5 for PATH_PXN
// Skipping 6 for PATH_PHB
#define LINK_SYS 7 //CPU之间连接的边
#define LINK_NET 8 //网络连接的边
extern const char* topoLinkTypeStr[];//连接类型字符串数组声明。

// Local (myself) 本地路径类型编号。
#define PATH_LOC 0 

// Connection traversing NVLink NVLink路径类型编号
#define PATH_NVL 1

// Connection through NVLink using an intermediate GPU 通过中间GPU的NVLink路径类型编号
#define PATH_NVB 2

// Connection traversing at most a single PCIe bridge 最多经过一个PCIe桥的路径类型编号。
#define PATH_PIX 3

// Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge) 经过多个PCIe桥但不经过主桥的路径类型编号
//PCIe Host Bridge 是连接 CPU（或主内存/主机控制器）与 PCIe 总线的桥接芯片，这里的意思就是不经过CPU。
#define PATH_PXB 4

// Connection between a GPU and a NIC using an intermediate GPU. Used to enable rail-local, aggregated network send/recv operations.
//GPU与NIC间通过中间GPU的路径类型编号
#define PATH_PXN 5

// Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU) 经过PCIe主桥（通常是CPU）的路径类型编号
#define PATH_PHB 6

// Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI) 经过PCIe和SMP互连（如QPI/UPI）的路径类型编号
#define PATH_SYS 7

// Connection through the network 
#define PATH_NET 8

// New type of path which should precede PATH_PIX
// PATH_PORT 虽然底层实现和 PATH_NVL 一致，
// 但它在某些上下文下（比如网络端口聚合、虚拟网卡建模等）强调的是“端口级别的聚合”或“虚拟设备的端口路径”，而不是单纯的物理 NVLink。
#define PATH_PORT PATH_NVL

// Disconnected 断开连接的路径类型编号
#define PATH_DIS 9
extern const char* topoPathTypeStr[];//路径类型字符串数组声明。

struct ncclTopoNode;//拓扑节点结构体
struct ncclTopoLink { //拓扑连接结构体，包含类型、带宽、远端节点指针。
  int type;
  float bw;
  struct ncclTopoNode* remNode;
};
// Allows for up to 32 NICs per node on GB200-NVL72
#define NCCL_TOPO_MAX_LINKS 576 //每个节点最大连接数576（支持多NIC）
#define NCCL_TOPO_MAX_HOPS (NCCL_TOPO_MAX_NODES*NCCL_TOPO_NODE_TYPES)//最大跳数=最大节点数*节点类型数。一个GPU节点，可能需要到所有NIC、所有CPU、所有PCI等类型节点建立通信路径。（每个都有256个）
//这里linklist其实才是一条路径，那么多个linklist就是多条路径
struct ncclTopoLinkList {
  struct ncclTopoLink* list[NCCL_TOPO_MAX_HOPS];//按顺序记录了一条路径上的边
  int count;//表示这条路径上总共经过了多少跳（即多少条边）。
  float bw;//这条路径的带宽（通常是路径上最小带宽的那一段，瓶颈带宽）。
  int type;//路径类型（如 PATH_NVL、PATH_PXB 等）。
};

#define NCCL_TOPO_CPU_INTEL_BDW 1
#define NCCL_TOPO_CPU_INTEL_SKL 2

#define NCCL_TOPO_UNDEF (-1)

#define NCCL_TOPO_ID_LOCAL_ID_MASK 0x00ffffffffffffff
#define NCCL_TOPO_ID_SYSTEM_ID(id) (id >> 56) //获取系统ID（高8位）
#define NCCL_TOPO_ID_LOCAL_ID(id) (id & NCCL_TOPO_ID_LOCAL_ID_MASK)//获取本地ID（低56位）。
#define NCCL_TOPO_LOCAL_NIC_ID(numaid, busid) (((int64_t)numaid << 56) + busid) //生成本地NIC的唯一ID（NUMA域+busid）。
#define NCCL_TOPO_ID(systemid, localid) (((int64_t)systemid << 56) + (localid & NCCL_TOPO_ID_LOCAL_ID_MASK))//生成全局唯一ID（系统ID+本地ID）

struct ncclTopoNode {
  int type;
  int64_t id;
  // Type specific data 类型特定数据
  union {
    struct {
      int dev; // NVML dev number
      int rank;
      int cudaCompCap;
      int gdrSupport;
    }gpu;
    struct {
      int dev; // Plugin dev number  插件设备编号（网络设备的唯一标识）。
      uint64_t asic;// 网络设备的ASIC芯片编号（唯一标识）。
      int port; // 端口号。
      float bw;// 网络带宽（GB/s）。
      float latency;// 网络延迟（单位：us或ns，具体看实现）。
      int gdrSupport;
      int collSupport;// 是否支持collective offload（集体通信加速）。
      int maxChannels;// 最大支持的通信通道数。
    }net;
    struct {
      int arch; // CPU架构
      int vendor;//CPU 厂商（如 Intel、AMD、兆芯等
      int model;//CPU 的具体型号（如 BDW、SKL、YONGFENG 等）
      cpu_set_t affinity;
    }cpu;
    struct {
      uint64_t device; // PCI设备号（用于唯一标识PCI设备）。
    }pci;
  };
  int nlinks; //连接数
  struct ncclTopoLink links[NCCL_TOPO_MAX_LINKS]; //连接数组，相当于节点的边
  // Pre-computed paths to GPUs and NICs 示“本节点到其它类型节点的预计算路径”（是路径，不是直接的边）。当前节点到每一种类型节点的“最优路径”集合
  // 每条路径是由一系列 ncclTopoLink* 组成，描述了从本节点出发，经过哪些边（links），最终到达目标类型节点。
  // 这里ncclTopoLinkList*其实就是多条路径的集合
  struct ncclTopoLinkList* paths[NCCL_TOPO_NODE_TYPES]; 
  // Used during search
  uint64_t used;
};

struct ncclTopoNodeSet {//节点集合
  int count;
  struct ncclTopoNode nodes[NCCL_TOPO_MAX_NODES];
};

struct ncclTopoSystem {
  int systemId; //系统ID
  uint64_t hostHashes[NCCL_TOPO_MAX_NODES]; //主机哈希
  int nHosts;
  struct ncclTopoNodeSet nodes[NCCL_TOPO_NODE_TYPES];//每种类型的节点集合。
  float maxBw;
  float totalBw;
};

ncclResult_t ncclTopoGetNode(struct ncclTopoSystem* system, struct ncclTopoNode** node, int type, uint64_t id);
ncclResult_t ncclTopoCreateNode(struct ncclTopoSystem* system, struct ncclTopoNode** node, int type, uint64_t id);
ncclResult_t ncclTopoRemoveNode(struct ncclTopoSystem* system, int type, int id);
ncclResult_t ncclTopoConnectNodes(struct ncclTopoNode* node, struct ncclTopoNode* remNode, int type, float bw);
ncclResult_t ncclTopoPrintPaths(struct ncclTopoSystem* system);
ncclResult_t ncclTopoLoadSystem(const char* xmlTopoFile, struct ncclTopoSystem* system);
ncclResult_t ncclTopoGetIntermediateRank(struct ncclTopoSystem* system, int rank, int64_t netId, int* intermediateRank);
ncclResult_t ncclTopoGetGpuMinPath(struct ncclTopoSystem* system, int type, int* min);
ncclResult_t ncclTopoGetGpuMaxPath(struct ncclTopoSystem* system, int type, int* max);

#define NCCL_TOPO_XML_MAX_NODES 256
#define NCCL_GRAPH_XML_MAX_NODES 4096
ncclResult_t ncclTopoGetSystemFromXml(struct ncclXml* xml, struct ncclTopoSystem** topoSystem, uint64_t localHostHash);
ncclResult_t ncclTopoGetGraphFromXml(struct ncclXmlNode *xmlGraphs, struct ncclTopoSystem* system, struct ncclTopoGraph* graph, int* nChannels);
ncclResult_t ncclTopoGetXmlFromGraphs(int ngraphs, struct ncclTopoGraph** graphs, struct ncclTopoSystem* system, struct ncclXml *xml);

ncclResult_t ncclTopoGetCompCap(struct ncclTopoSystem* system, int* ccMin, int* ccMax);

static ncclResult_t ncclTopoIdToIndex(struct ncclTopoSystem* system, int type, int64_t id, int* index) {
  *index = -1;
  for (int i=0; i<system->nodes[type].count; i++) {
    if (system->nodes[type].nodes[i].id == id) {
      *index = i;
      return ncclSuccess;
    }
  }
  return ncclInternalError;
}

static ncclResult_t ncclTopoRankToIndex(struct ncclTopoSystem* system, int rank, int* index) {
  *index = -1;
  for (int i=0; i<system->nodes[GPU].count; i++) {
    if (system->nodes[GPU].nodes[i].gpu.rank == rank) {
      *index = i;
      return ncclSuccess;
    }
  }
  return ncclInternalError;
}

static ncclResult_t ncclTopoDevToRank(struct ncclTopoSystem* system, int dev, int* rank) {
  *rank = -1;
  for (int i=0; i<system->nodes[GPU].count; i++) {
    if (NCCL_TOPO_ID_SYSTEM_ID(system->nodes[GPU].nodes[i].id) != system->systemId) continue; // Only consider GPUs on our node
    if (system->nodes[GPU].nodes[i].gpu.dev == dev) {
      *rank = system->nodes[GPU].nodes[i].gpu.rank;
      return ncclSuccess;
    }
  }
  return ncclInternalError;
}

static ncclResult_t ncclTopoIdToNetDev(struct ncclTopoSystem* system, int64_t id, int* netDev) {
  *netDev = -1;
  for (int i=0; i<system->nodes[NET].count; i++) {
    if (system->nodes[NET].nodes[i].id == id) {
      *netDev = system->nodes[NET].nodes[i].net.dev;
      return ncclSuccess;
    }
  }
  WARN("Could not find NET with id %lx", id);
  return ncclInternalError;
}

// Returns NVLink bw in GB/s
static float ncclTopoNVLinkBw(int cudaCompCap) {
  return
    cudaCompCap >= 100 ? SM100_NVLINK_BW :
    cudaCompCap >= 90 ? SM90_NVLINK_BW :
    cudaCompCap == 86 ? SM86_NVLINK_BW :
    cudaCompCap >= 80 ? SM80_NVLINK_BW :
    cudaCompCap >= 70 ? SM70_NVLINK_BW :
    cudaCompCap >= 60 ? SM60_NVLINK_BW :
    SM80_NVLINK_BW;
}

// Mirror bits
static bool isPow2(int val) {
  return (val & (val-1)) == 0;
}
//按位镜像（位反转）一个整数的低若干位 pow2 ：必须是 2 的幂，表示要镜像的位数（比如 8 表示低 3 位，16 表示低 4 位
//例如， val=3 （ 011 ）， pow2=8 （3 位），镜像后是 110 （即 6） 这种操作常用于环形、树形等通信模式下的对称索引变换。
static int mirrorBits(int val, int pow2) {
  int mirror = 0;
  for (int b=1, mb=(pow2>>1); b<pow2; b<<=1, mb>>=1) if (val & b) mirror |= mb;
  return mirror;
}
#endif
