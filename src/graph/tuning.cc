/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "device.h"
#include "comm.h"
#include "topo.h"

NCCL_PARAM(Nthreads, "NTHREADS", -2);
NCCL_PARAM(Ll128Nthreads, "LL128_NTHREADS", -2);
//把给定的线程数限制min和max之间，如果env小于0，则设置为默认值
static int getNthreads(const char* name, int env, int min, int max, int def) {
  int nt = env;
  if (nt > 0) {
    if (nt % WARP_SIZE != 0) {
      WARN("Invalid %s %d (must be a multiple of %d)", name, nt, WARP_SIZE);
      nt = max;
    } else if (nt > max) {
      WARN("Invalid %s %d (maximum %d).", name, nt, max);
      nt = max;
    } else if (nt < min) {
      WARN("Invalid %s %d (minimum %d).", name, nt, min);
      nt = min;
     }
  } else {
    nt = def;
  }
  return nt;
}

// Parse a map of prefixes to a list of elements. The first prefix is
// optional and, if not present, the list of elements will be applied
// to all prefixes. Only the first list of elements can lack a
// prefix. Prefixes (if present) are followed by a colon. Lists of
// elements are comma delimited. Mappings of prefix to the lists of
// elements are semi-colon delimited.
//
// For example:
//
//     NCCL_ALGO="ring,collnetdirect;allreduce:tree,collnetdirect;broadcast:ring"
// Enable ring and collnetdirect for all functions, then select tree
// and collnetdirect for allreduce and ring for broadcast.
//
//     NCCL_PROTO="LL,Simple;allreduce:^LL"
// Enable LL and Simple for all functions, but everything except LL
// for allreduce.
//
//     NCCL_PROTO="^LL128;allreduce:LL128"
// Enable everything but LL128, but only LL128 for allreduce.
ncclResult_t parseList(const char* str, const char* prefixElems[], int nprefixes, const char* elems[], int nelems, int* list) {
  char* fullStr = strdup(str);
  char* tmpFullStr;
  char* fullToken = strtok_r(fullStr, ";", &tmpFullStr);
  while (fullToken) {
    char* subToken = strdup(fullToken);
    char* tmpSubStr;
    char* prefix = strtok_r(subToken, ":", &tmpSubStr);
    char* elemList = strtok_r(NULL, ":", &tmpSubStr);
    if (elemList == NULL) {
      if (fullToken != fullStr) {
        // It makes no sense for any entry other than the first to not have a prefix,
        // because then all the prefixes before the prefix-less entry would be
        // overwritten.
        WARN("All entries except the first must have a prefix: \"%s\"", str);
        return ncclInvalidUsage;
      }
      elemList = prefix;
      prefix = NULL;
    }

    int unset, set;
    if (elemList[0] == '^') {
      unset = 1; set = 0; elemList++;
    } else {
      unset = 0; set = 1;
    }

    bool foundPrefix = false;
    for (int p=0; p<nprefixes; p++) {
      if (prefix && strcasecmp(prefix, prefixElems[p]) != 0) continue;
      foundPrefix = true;
      for (int e=0; e<nelems; e++) list[p*nelems+e] = unset;

      char* tokStr = strdup(elemList);
      char* tmpStr;
      char* elem = strtok_r(tokStr, ",", &tmpStr);
      while (elem) {
        int e;
        for (e=0; e<nelems; e++) {
          if (strcasecmp(elem, elems[e]) == 0) {
            list[p*nelems+e] = set;
            break;
          }
        }
        if (e==nelems) {
          WARN("Unrecognized element token \"%s\" when parsing \"%s\"", elem, str);
          return ncclInvalidUsage;
        }
        elem = strtok_r(NULL, ",", &tmpStr);
      }
      free(tokStr);
    }
    if (!foundPrefix) {
      WARN("Unrecognized prefix token \"%s\" when parsing \"%s\"", prefix, str);
      return ncclInvalidUsage;
    }
    free(subToken);

    fullToken = strtok_r(NULL, ";", &tmpFullStr);
  }
  free(fullStr);
  return ncclSuccess;
}

// Latencies in us, Bandwidths in GB/s
// Tree { LL, LL128, Simple } , Ring { LL, LL128, Simple }
static const float baseLat  [NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS] = {
       {  6.8, 14.0,  8.4 }, {  6.6, 14.0,  8.4 },  // Tree, Ring
       {    0,    0,    0 }, {    0,    0,    0 },  // Collnet Direct, Chain
       {    0,    0,    0 }, {    0,    0,    0 }}; // NVLS, NVLS Tree

// NVLink, PCI, Network
#define NCCL_HW_NVLINK 0
#define NCCL_HW_PCI 1
#define NCCL_HW_NET 2
static float hwLat [3][NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS] =
{ /* NVLINK */
  { /* Tree (LL/LL128/Simple)*/ { .6, 1.25, 4.0 }, /* Ring (LL/LL128/Simple)*/ { .6, 1.9, 3.4 },
    /* CollNetDirect (Simple)*/ { 0, 0, 3.7 }, /* CollNetChain (Simple)*/ { 0, 0, 2.8 },
    /* NVLS */ { 0, 0, 25 }, /* NVLSTree */ { 0, 0, 25 } },
  /* PCI */
  { /* Tree (LL/LL128/Simple)*/ { 1.0, 1.9, 4.0 }, /* Ring (LL/LL128/Simple)*/ { 1.0, 2.5, 5.7 },
    /* CollNetDirect (Simple)*/ { 0, 0, 3.7 }, /* CollNetChain (Simple)*/ { 0, 0, 2.8 },
    /* NVLS */ { 0, 0, 0 }, /* NVLSTree */ { 0, 0, 0 } },
  /* NET */
  { /* Tree (LL/LL128/Simple)*/ { 5.0, 8.5, 14 }, /* Ring (LL/LL128/Simple)*/ { 2.7, 4.0, 14.0 },
    /* CollNetDirect (Simple)*/ { 0, 0, 31 }, /* CollNetChain (Simple)*/ { 0, 0, 30 },
    /* NVLS */ { 0, 0, 18 }, /* NVLSTree */ { 0, 0, 14 } }
};

/* Array indexes used below */
#define VOLTA_COMPCAP_IDX 0
#define AMPERE_COMPCAP_IDX 1
#define HOPPER_COMPCAP_IDX 2
#define BLACKWELL_COMPCAP_IDX 3

// LL128 max BW per channel
// 这里的N是指Node，所以N1、N2是指node数量，以volta为例：单节点（N1）和双节点（N2）：最大带宽为 39.0 GB/s，四节点及以上（N4）：带宽下降到 20.4 GB/s
static const double llMaxBws[][3] = {
  /* Volta-N1/Intel-N2/Intel-N4) */ {39.0, 39.0, 20.4},
  /* Ampere-N1/AMD-N2/AMD-N4) */ {87.7, 22.5 /*avg of ring & tree*/, 19.0},
  /* Hopper-N1/AMD-N2/AMD-N4) */ {141.0, 45.0 /*avg of ring & tree*/, 35.0},
  /* Blackwell-N1/AMD-N2/AMD-N4) */ {2*141.0, 2*45.0 /*avg of ring & tree*/, 2*35.0},
};

static const double perChMaxRingLL128Bws[][3] = {
  /* Volta (N1/N2/N4) */  {20.0, 20.0, 20.0},
  /* Ampere (N1/N2/N4) */ {20.0, 20.0, 20.0},
  /* Hopper (N1/N2/N4) */ {36.7, 36.7, 36.7},
  /* Blackwell (N1/N2/N4) */ {2*36.7, 2*36.7, 2*36.7},
};
static const double perChMaxTreeLL128Bws[][3] = {
  /* Volta (N1/N2/N4) */  {20.0, 20.0, 20.0},
  /* Ampere (N1/N2/N4) */ {20.0, 20.0, 20.0},
  /* Hopper (N1/N2/N4) */ {36.7, 36.7, 29.0},
  /* Blackwell (N1/N2/N4) */ {2*36.7, 2*36.7, 2*29.0},
};
static const double perChMaxTreeBws[][3] = {
  /* Volta (N1/N2/N4) */  {26.5, 18.5, 10.0},
  /* Ampere (N1/N2/N4) */ {24.0, 23.6, 17.8},
  /* Hopper (N1/N2/N4) */ {38.7, 41.4, 36.0},
  /* Blackwell (N1/N2/N4) */ {2*38.7, 2*41.4, 2*36.0},
};

NCCL_PARAM(PatEnable, "PAT_ENABLE", 2);
static int ncclPatEnable(struct ncclComm* comm) {
  int patEnable = ncclParamPatEnable();
  if (patEnable != 2) return patEnable;
  if (comm->nNodes != comm->nRanks) return 0; // PAT only supports 1 GPU per node
  if (comm->netDeviceType != NCCL_NET_DEVICE_HOST) return 0;   // PAT doesn't support net device offload
  return 1;
}

// Network post overhead in ns (1000 = 1 us)
NCCL_PARAM(NetOverhead, "NET_OVERHEAD", -2);

static float getNetOverhead(struct ncclComm* comm) {
  if (ncclParamNetOverhead() != -2) return ncclParamNetOverhead() * .001;
  if (comm->cpuArch == NCCL_TOPO_CPU_ARCH_X86 && comm->cpuVendor == NCCL_TOPO_CPU_VENDOR_INTEL) return 1.0;
  if (comm->cpuArch == NCCL_TOPO_CPU_ARCH_X86 && comm->cpuVendor == NCCL_TOPO_CPU_VENDOR_AMD) return 2.0;
  return 1.0;
}
/*
基于硬件拓扑、GPU 架构、网络带宽等信息，预测每种通信操作在不同算法和协议下的执行时间（延迟 + 带宽），从而选择最优的通信策略。

*/
ncclResult_t ncclTopoTuneModel(struct ncclComm* comm, int minCompCap, int maxCompCap, struct ncclTopoGraph** graphs) {
  //设置 Ring + Simple 协议的最大线程数，通过环境变量或默认值获取。
  int simpleDefaultThreads = (graphs[NCCL_ALGO_RING]->bwIntra*graphs[NCCL_ALGO_RING]->nChannels <= PCI_BW) ? 256 : NCCL_SIMPLE_MAX_NTHREADS;
  comm->maxThreads[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE] =
    getNthreads("NCCL_NTHREADS", ncclParamNthreads(), 2*WARP_SIZE, NCCL_SIMPLE_MAX_NTHREADS, simpleDefaultThreads);
  //设置 Tree + Simple 协议的最大线程数。
    comm->maxThreads[NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE] =
    getNthreads("NCCL_NTHREADS", ncclParamNthreads(), 2*WARP_SIZE, NCCL_SIMPLE_MAX_NTHREADS, NCCL_SIMPLE_MAX_NTHREADS);
  //CollNetDirect、CollNetChain、NVLS、NVLSTree + Simple 协议使用最大线程数 NCCL_MAX_NTHREADS。
    comm->maxThreads[NCCL_ALGO_COLLNET_DIRECT][NCCL_PROTO_SIMPLE] =
    comm->maxThreads[NCCL_ALGO_COLLNET_CHAIN][NCCL_PROTO_SIMPLE] =
    comm->maxThreads[NCCL_ALGO_NVLS][NCCL_PROTO_SIMPLE] =
    comm->maxThreads[NCCL_ALGO_NVLS_TREE][NCCL_PROTO_SIMPLE] = NCCL_MAX_NTHREADS;
    //Ring 和 Tree 算法在 LL 协议下使用最大线程数
  comm->maxThreads[NCCL_ALGO_RING][NCCL_PROTO_LL] = comm->maxThreads[NCCL_ALGO_TREE][NCCL_PROTO_LL] =
    getNthreads("NCCL_NTHREADS", ncclParamNthreads(), 2*WARP_SIZE, NCCL_LL_MAX_NTHREADS, NCCL_LL_MAX_NTHREADS);
    // Ring 和 Tree 算法在 LL128 协议下使用最大线程数
    comm->maxThreads[NCCL_ALGO_RING][NCCL_PROTO_LL128] = comm->maxThreads[NCCL_ALGO_TREE][NCCL_PROTO_LL128] =
    getNthreads("NCCL_LL128_NTHREADS", ncclParamLl128Nthreads(), NCCL_LL128_MAX_NTHREADS/4, NCCL_LL128_MAX_NTHREADS, NCCL_LL128_MAX_NTHREADS);

  int nNodes = comm->nNodes;
  int nRanks = comm->nRanks;
  if (nRanks <= 1) return ncclSuccess;//单卡无需调优。
  //根据最低 Compute Capability 决定使用的索引，用于查找对应架构下的带宽数据。
  int compCapIndex = minCompCap >= 100 ? BLACKWELL_COMPCAP_IDX : (minCompCap >= 90 ? HOPPER_COMPCAP_IDX : minCompCap >= 80 ? AMPERE_COMPCAP_IDX : VOLTA_COMPCAP_IDX);
  int index2 = nNodes <= 2 ? nNodes-1 : 2;//节点数小于等于 2 时用 (nNodes - 1)，否则固定为 2。
  // LL: for single node, we look at GPU type; for multi-node, we look at CPU type
  //单节点时使用 GPU 类型索引；多节点且 CPU 是 AMD 或混合类型时使用索引 1，否则索引 0。
  //这是因为多节点时，数据传输主要受到网络的影响，而cpu类型可能会影响网络性能。因此根据cpu厂商来选取
  int index1 = nNodes == 1 ? compCapIndex :
               (comm->cpuVendor == NCCL_TOPO_CPU_VENDOR_AMD || comm->cpuVendor == NCCL_TOPO_CPU_VENDOR_MIXED) ? 1 : 0;
  double llMaxBw = llMaxBws[index1][index2];
  double perChMaxTreeBw = perChMaxTreeBws[compCapIndex][index2];
  double perChMaxRingLL128Bw = perChMaxRingLL128Bws[compCapIndex][index2];
  double perChMaxTreeLL128Bw = perChMaxTreeLL128Bws[compCapIndex][index2];
  // De-penalize Tree/Simple latency on Power systems to favor Tree than Ring
  if (comm->cpuArch == NCCL_TOPO_CPU_ARCH_POWER) hwLat[NCCL_HW_PCI][NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE] = hwLat[NCCL_HW_PCI][NCCL_ALGO_RING][NCCL_PROTO_SIMPLE];
  float ppn = (float)nRanks / nNodes;//每个节点上的进程数（Process Per Node）。

  // 初始化硬件类型数组,也就是节点内连接类型和节点间连接类型。
  int intraHw[NCCL_NUM_ALGORITHMS], hw[NCCL_NUM_ALGORITHMS];
  for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) intraHw[a] = graphs[a]->typeIntra == LINK_NVL ? NCCL_HW_NVLINK : NCCL_HW_PCI;
  for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) hw[a] = nNodes == 1 ? intraHw[a] : NCCL_HW_NET;

  //遍历所有通信操作(AllReduce、AllGather 等)
  for (int coll=0; coll<NCCL_NUM_FUNCTIONS; coll++) {
    //步骤数 AllReduce: 2×(nRanks - 1),ReduceScatter/AllGather: nRanks - 1，其他算法nRanks
    int nsteps = coll == ncclFuncAllReduce ? 2*(nRanks-1) :
      coll == ncclFuncReduceScatter || coll == ncclFuncAllGather ? nRanks-1 :
      nRanks;

    for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
      //过滤掉不适用于当前通信类型的算法（如 Broadcast 只能用 Ring 算法）。
      if ((coll == ncclFuncBroadcast || coll == ncclFuncReduce) && a != NCCL_ALGO_RING) continue;
      //这里的意思是，如果当前通信操作是 ReduceScatter 或 AllGather，并且算法不是 PAT、Ring、NVLS 或 CollNet Direct，则跳过。
      //(按照这个意思是，这些算法不支持 ReduceScatter 或 AllGather 操作。但是我目前还是比较疑惑，需要后续看具体的实现来判断)
      if ((coll == ncclFuncReduceScatter || coll == ncclFuncAllGather)
          && a != NCCL_ALGO_PAT && a != NCCL_ALGO_RING
          && a != NCCL_ALGO_NVLS && a != NCCL_ALGO_COLLNET_DIRECT) continue;
      if (coll == ncclFuncAllReduce && a == NCCL_ALGO_PAT) continue;
      //遍历所有协议（LL、LL128、Simple 等）
      for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
        //NVLS/NVLS_Tree 只支持 Simple 协议。
        if ((a == NCCL_ALGO_NVLS || a == NCCL_ALGO_NVLS_TREE) && p != NCCL_PROTO_SIMPLE) continue;
        //PAT 算法只在 Simple 协议且启用时才可用。
        if ((coll == ncclFuncReduceScatter || coll == ncclFuncAllGather)
            && a == NCCL_ALGO_PAT && (p != NCCL_PROTO_SIMPLE || ncclPatEnable(comm) == 0)) continue;
        
        int collnet = (a == NCCL_ALGO_COLLNET_DIRECT || a == NCCL_ALGO_COLLNET_CHAIN) ? 1 : 0;
        //判断是节点内还是节点间通信，并选择相应带宽。在现代数据中心中，两个节点之间可能通过NVLink Switch、PCIe 直接连接
        float bw = nNodes <= 2 || collnet ? graphs[a]->bwIntra : graphs[a]->bwInter;
        if (a == NCCL_ALGO_NVLS) bw = std::min(graphs[a]->bwIntra, graphs[a]->bwInter);
        if (a == NCCL_ALGO_NVLS_TREE) bw = std::min(graphs[a]->bwIntra, nNodes <= 2 ? graphs[a]->bwInter : graphs[a]->bwInter/2);
        float busBw = graphs[a]->nChannels * bw;

        // Various model refinements 带宽调整（根据不同算法/协议优化）
        //Ring + LL：限制bus最大带宽并打五折。
        if (a == NCCL_ALGO_RING && p == NCCL_PROTO_LL) { busBw = std::min(llMaxBw, busBw * .5); }
        //Ring + LL128：乘以修正系数 0.92 并取最大值。
        if (a == NCCL_ALGO_RING && p == NCCL_PROTO_LL128) busBw = std::min(busBw * (0.92 /*120.0/128.0*/), graphs[a]->nChannels*perChMaxRingLL128Bw);
        //Tree + AllReduce：乘以 0.92 并取最大值。
        if (a == NCCL_ALGO_TREE && coll == ncclFuncAllReduce) busBw = std::min(busBw*.92, graphs[a]->nChannels*perChMaxTreeBw);
        //Tree + LL：除以 3.8 并取最大值。
        if (a == NCCL_ALGO_TREE && p == NCCL_PROTO_LL) busBw = std::min(busBw*1.0/3.8, llMaxBw);
        //Tree + LL128：乘以特定比例因子（单节点 7/9，多节点 120/128）并取最大值。
        if (a == NCCL_ALGO_TREE && p == NCCL_PROTO_LL128) busBw = std::min(busBw * (nNodes == 1 ? 7.0/9.0 : 120.0/128.0), graphs[a]->nChannels*perChMaxTreeLL128Bw);
        //Tree 模式下调低 15% 带宽。
        if (a == NCCL_ALGO_TREE && graphs[a]->pattern == NCCL_TOPO_PATTERN_TREE) busBw *= .85;
        //PAT 算法带宽打七五折。
        if (a == NCCL_ALGO_PAT) busBw *= .75;
        if (a == NCCL_ALGO_COLLNET_DIRECT && p != NCCL_PROTO_SIMPLE) busBw = 0;  // Not used
        if (a == NCCL_ALGO_COLLNET_CHAIN && p != NCCL_PROTO_SIMPLE) busBw = 0;  // Not used
        
        //AllGather/ReduceScatter：按 GPU/NIC 比例缩放带宽。
        if (a == NCCL_ALGO_COLLNET_DIRECT && p == NCCL_PROTO_SIMPLE) {
          if (coll == ncclFuncAllGather || coll == ncclFuncReduceScatter) {
            busBw = ppn * bw;
            // AllGather/ReduceScatter requires 1:1 GPU:NIC 
            int nicPerNode = comm->collNetHeadsNum;//每个节点 NIC 数量。
            //AllGather 多节点时检查 CollNet 是否支持。也就是要支持collnet，并且rank数小于nic数量
            if (coll == ncclFuncAllGather && comm->nNodes > 1) {
              if (!comm->ncclCollNet || !comm->ncclCollNet->iallgather || ppn > nicPerNode) busBw = 0;
            }
            //ReduceScatter 多节点时检查 CollNet 是否支持。也就是要支持collnet，并且rank数小于nic数量
            if (coll == ncclFuncReduceScatter && comm->nNodes > 1) {
              if (!comm->ncclCollNet || !comm->ncclCollNet->ireducescatter || ppn > nicPerNode) busBw = 0;
            }
            // Measured corrective ratio needed at 1 ppn and 8ppn. Here we hackishly
            // interpolate the two. "在1 PPN和8 PPN的配置下测量得到修正比率，
            // 并通过简单的线性插值估算中间PPN（如2-7 PPN）所需的修正比率。"​
            // 插值调整带宽（1 ppn 到 8 ppn 之间插值）。
            float w = (ppn-1)/(8-1);
            busBw *= w*0.85 + (1-w)*0.95;
          } else { //CollNetDirect：根据 GPU/NIC 比例调整带宽。
            // Collnet+Direct requires all GPUs to have a local NIC to work at full speed
            //graphs[a]->nChannels：当前算法使用的通道数量（Channel Count），通常也代表可用的 NIC 数量（因为 CollNetDirect 要求每个 Channel 对应一个 NIC）。
            float factor = ppn / (1.0*graphs[a]->nChannels); // GPU/NIC ratio
            factor -= (factor-1)/2; //这一步是对 factor 做了一个经验性的“软化”处理，目的是当 GPU/NIC 比例较高时，降低惩罚力度，避免过度削减带宽估计值。
            busBw /= factor; //根据上面计算出的 factor，将原始带宽除以这个值，模拟因 NIC 不足导致的带宽下降。
            if (minCompCap >= 90) busBw *= .85;
          }
        }

        // Convert bus BW to algorithm BW
        if (!(a != NCCL_ALGO_RING && (coll == ncclFuncAllGather || coll == ncclFuncReduceScatter))) {
          float ratio = 1.0f;
          if (a == NCCL_ALGO_RING) ratio *= (1.0 * nRanks) / nsteps;//这个计算推导一下就知道了，可以推导时间的比值，从而算得带宽比值
          //NVLS 是全互联架构，但受硬件限制，带宽打八三折
          else if (a == NCCL_ALGO_NVLS || a == NCCL_ALGO_NVLS_TREE) ratio *= 5.0/6.0;
          else ratio *= .5;//默认打五折，表示非 Ring 类型算法效率较低
          busBw *= ratio;
        }
        comm->bandwidths[coll][a][p] = busBw;
        comm->latencies[coll][a][p] = baseLat[a][p];
        float intraLat = hwLat[intraHw[a]][a][p];//节点内通信延迟
        // With ppn=1 latencies are fully exposed, use the Tree network latency
        // 当 ppn == 1（每个节点只有一个 GPU）时，跨节点延迟统一使用 Tree 算法的延迟。
        float interLat = ppn == 1 ? hwLat[NCCL_HW_NET][NCCL_ALGO_TREE][p] : hwLat[NCCL_HW_NET][a][p];
        interLat += graphs[a]->latencyInter;
        // Also add the flush extra latency 对 Simple 协议，额外加一次 latencyInter，模拟 flush 数据的开销。
        if (p == NCCL_PROTO_SIMPLE) interLat += graphs[a]->latencyInter;

        //根据不同算法细化延迟计算
        if (a == NCCL_ALGO_RING) {
          float lat = hwLat[hw[a]][a][p];
          if ((coll == ncclFuncReduce || coll == ncclFuncBroadcast)) {
            if (graphs[a]->sameChannels) {//如果通道相同（sameChannels），直接加上基础延迟。这里可能的解释是，通道相同意味着路径固定、可复用。减少了开销。
              comm->latencies[coll][a][p] += lat;
            } else {//否则按 nsteps * lat 计算（Simple 协议用 Tree 的延迟替代）。
              if (p == NCCL_PROTO_SIMPLE) lat = hwLat[hw[a]][NCCL_ALGO_TREE][p]; // Add some chunk latency, waiting for proper chunk modeling
              comm->latencies[coll][a][p] += nsteps*lat;
            }
          } else {
            // Inter-node rings still have to launch nsteps * net overhead.
            /*
            netOverhead：网络通信的额外开销（比如启动通信、上下文切换、序列化等）
            在 Simple 协议中，由于缺乏像 LL/LL128 那样的高效 chunking 机制，这些开销会被放大。
            所以 NCCL 给 Simple 协议的 netOverhead 乘以 3，模拟其更高的启动成本。
            */
            float netOverhead = 0.0;
            if (nNodes > 1) {
              netOverhead = getNetOverhead(comm);//每个通信步骤中，涉及 NIC、DMA、CPU 调度等的最小启动延迟。
              if (p == NCCL_PROTO_SIMPLE) netOverhead *= 3;
            }
            intraLat = std::max(intraLat, netOverhead);
            int nInterSteps = nNodes == 1 ? 0 : coll == ncclFuncAllReduce ? 2*(nNodes-1) : nNodes-1;
            //节点内和节点间。
            comm->latencies[coll][a][p] += (nsteps-nInterSteps)*intraLat + nInterSteps*interLat;
          }
        } else if (a == NCCL_ALGO_TREE) {
          if (coll == ncclFuncAllReduce) {
            comm->latencies[coll][a][p] +=
              2 * ((nRanks/nNodes-1) * intraLat + log2i(nNodes) * interLat);
          }
        } else if (a == NCCL_ALGO_COLLNET_DIRECT) {
          comm->latencies[coll][a][p] +=
            2 * (std::min(1, (nRanks/nNodes-1)) * intraLat + (nRanks/nNodes-1) * 0.4) + interLat;  // Add 0.4 us arity serialization latency
        } else if (a == NCCL_ALGO_COLLNET_CHAIN) {
          comm->latencies[coll][a][p] += 2 * (nRanks/nNodes-1) * intraLat + interLat;
        } else if (a == NCCL_ALGO_NVLS) {
          comm->latencies[coll][a][p] = intraLat;
          if (nNodes > 1) comm->latencies[coll][a][p] += interLat;
        } else if (a == NCCL_ALGO_NVLS_TREE) {
          comm->latencies[coll][a][p] += intraLat + 2 * log2i(nNodes) * interLat;
        } else if (a == NCCL_ALGO_PAT) {
          if (coll == ncclFuncAllGather || coll == ncclFuncReduceScatter) {
            comm->latencies[coll][a][p] = 8 // Base time
              + log2i(nNodes) * (interLat/3.5) // Log latency
              + nRanks * 2.8; // Still a linear part; hopefully we'll manage to remove it at some point.
          }
        }
      }
    }
  }

  // Protocols/Algorithms enable/disable, and user overrides.
  // All are enabled except ll128 which is enabled by default only in certain cases.
  int protoEnable[NCCL_NUM_FUNCTIONS*NCCL_NUM_PROTOCOLS];
  int algoEnable[NCCL_NUM_FUNCTIONS*NCCL_NUM_ALGORITHMS];
  for (int f=0; f<NCCL_NUM_FUNCTIONS; f++) {
    for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
      protoEnable[f*NCCL_NUM_PROTOCOLS+p] = p == NCCL_PROTO_LL128 ? 2 : 1;
    }
    for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
      algoEnable[f*NCCL_NUM_ALGORITHMS+a] = 1;
    }
  }

  const char *protoStr = ncclGetEnv("NCCL_PROTO");
  if (protoStr) {
    INFO(NCCL_ENV, "NCCL_PROTO set by environment to %s", protoStr);
    //parseList 函数解析字符串格式，决定哪些协议（如 LL, Simple, LL128）在哪些操作（如 allreduce, broadcast）上被启用或禁用。
    NCCLCHECK(parseList(protoStr, ncclFuncStr, NCCL_NUM_FUNCTIONS, ncclProtoStr, NCCL_NUM_PROTOCOLS, protoEnable));
  }
  const char *algoStr = ncclGetEnv("NCCL_ALGO");
  if (algoStr) {
    INFO(NCCL_ENV, "NCCL_ALGO set by environment to %s", algoStr);
    NCCLCHECK(parseList(algoStr, ncclFuncStr, NCCL_NUM_FUNCTIONS, ncclAlgoStr, NCCL_NUM_ALGORITHMS, algoEnable));
  }

  //如果当前进程是 rank 0 并且设置了 NCCL_PROTO 或 NCCL_ALGO，则构建一个字符串，展示每个函数（Function）、协议（Protocol）、算法（Algorithm）的启用状态。
  if (comm->rank == 0 && (algoStr||protoStr)) {
    constexpr int strLength = 1024;
    char funcAlgoProtoTuningStr[strLength];
    int offset = 0;
    offset += snprintf(funcAlgoProtoTuningStr+offset, std::max(0, strLength-offset), "\n     Function | ");
    for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
      offset += snprintf(funcAlgoProtoTuningStr+offset, std::max(0, strLength-offset), "%8s  ", ncclProtoStr[p]);
    }
    offset += snprintf(funcAlgoProtoTuningStr+offset, std::max(0, strLength-offset), " | ");
    for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
      offset += snprintf(funcAlgoProtoTuningStr+offset, std::max(0, strLength-offset), "%13s  ", ncclAlgoStr[a]);
    }
    offset += snprintf(funcAlgoProtoTuningStr+offset, std::max(0, strLength-offset), "\n");

    for (int f=0; f<NCCL_NUM_FUNCTIONS; f++) {
      offset += snprintf(funcAlgoProtoTuningStr+offset, std::max(0, strLength-offset), "%13s | ", ncclFuncStr[f]);
      for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
        offset += snprintf(funcAlgoProtoTuningStr+offset, std::max(0, strLength-offset), "%8d  ", protoEnable[f*NCCL_NUM_PROTOCOLS+p]);
      }
      offset += snprintf(funcAlgoProtoTuningStr+offset, std::max(0, strLength-offset), " | ");
      for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
        offset += snprintf(funcAlgoProtoTuningStr+offset, std::max(0, strLength-offset), "%13d  ", algoEnable[f*NCCL_NUM_ALGORITHMS+a]);
      }
      offset += snprintf(funcAlgoProtoTuningStr+offset, std::max(0, strLength-offset), "\n");
    }

    INFO(NCCL_ENV, "Enabled NCCL Func/Proto/Algo Matrix:%s", funcAlgoProtoTuningStr);
  }

  int nvsCount = 0;
  NCCLCHECK(ncclTopoGetNvsCount(comm->topo, &nvsCount));
//基于硬件条件自动禁用某些算法
  for (int f=0; f<NCCL_NUM_FUNCTIONS; f++) {
    for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
      int disable = 0;
      // Disable NVLS Tree on a single node 例如，在单节点系统中禁用 NVLS Tree，因为该算法适用于多节点拓扑。
      if (comm->nNodes == 1 && a == NCCL_ALGO_NVLS_TREE) disable = 1;
      // Disable Collnet+Direct, Collnet+Chain or Collnet+NVLS if collnet is not supported.
      if (comm->collNetSupport == 0 &&
          (a == NCCL_ALGO_COLLNET_DIRECT ||
           a == NCCL_ALGO_COLLNET_CHAIN ||
           (a == NCCL_ALGO_NVLS && comm->nNodes > 1))) disable = 1;
      // Disable CollNet+Direct if not on an NVSwitch system
      if (nvsCount == 0 && a == NCCL_ALGO_COLLNET_DIRECT) disable = 1;
      if (disable) algoEnable[f*NCCL_NUM_ALGORITHMS+a] = 0;
    }
  }
  //如果启用了 LL128 协议，则进一步检查是否满足特定的硬件要求（如是否使用 Volta/Ampere 架构、是否通过 NVLink 连接等）。
  for (int c=0; c<NCCL_NUM_FUNCTIONS; c++) for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    int pEnable = protoEnable[c*NCCL_NUM_PROTOCOLS+p];
    if (pEnable == 2 && p == NCCL_PROTO_LL128) {
      // Enable LL128 by default only on Volta/Ampere/Hopper/Blackwell+NVLink. Other cases are not tested and may cause silent data corruption.
      pEnable = 1;
      pEnable &= (graphs[a]->typeInter <= PATH_PXB || (minCompCap >= 90 && graphs[a]->typeInter <= PATH_PXN));
      pEnable &= (graphs[a]->typeIntra <= PATH_NVB);
      pEnable &= (minCompCap == maxCompCap);
      switch (minCompCap) {
      case 70: pEnable &= 1; break;
      case 80: pEnable &= 1; break;
      case 90: pEnable &= !(CUDART_VERSION == 11080 && c == ncclFuncAllReduce && a == NCCL_ALGO_RING && comm->nRanks == 2); break;
      case 100: pEnable &= 1; break;
      case 120: pEnable &= 1; break;
      default: pEnable &= 0; break;
      }
    }
    if (pEnable == 0) comm->bandwidths[c][a][p] = 0;
    if (algoEnable[c*NCCL_NUM_ALGORITHMS+a] == 0) comm->bandwidths[c][a][p] = 0;
  }
  //打印算法/协议性能信息
  if (comm->rank == 0) {
    constexpr int lineLen = 1024;
    char line[lineLen];
    int offset = 0;
    for (int block=0; block<DIVUP(NCCL_NUM_ALGORITHMS, 3); block++) {
      offset = snprintf(line, lineLen, "  Algorithm   |");
      for (int ba=0; ba<3; ba++) {
        int a = block*3+ba;
        if (a >= NCCL_NUM_ALGORITHMS) continue;
        offset += snprintf(line+offset, std::max(0, lineLen-offset), " %14s   %14s   %14s |", "", ncclAlgoStr[a], "");
      }
      INFO(NCCL_TUNING, "%s", line);
      offset = snprintf(line, lineLen, "  Protocol    |");
      for (int ba=0; ba<3; ba++) {
        for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
          offset += snprintf(line+offset, std::max(0, lineLen-offset), " %14s |", ncclProtoStr[p]);
        }
      }
      INFO(NCCL_TUNING, "%s", line);
      offset = snprintf(line, lineLen, " Max NThreads |");
      for (int ba=0; ba<3; ba++) {
        int a = block*3+ba;
        if (a >= NCCL_NUM_ALGORITHMS) continue;
        for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
          offset += snprintf(line+offset, std::max(0, lineLen-offset), " %14d |", comm->maxThreads[a][p]);
        }
      }
      INFO(NCCL_TUNING, "%s", line);
      for (int c=0; c<NCCL_NUM_FUNCTIONS; c++) {
        offset = snprintf(line, lineLen, "%13s |", ncclFuncStr[c]);
        for (int ba=0; ba<3; ba++) {
          int a = block*3+ba;
          if (a >= NCCL_NUM_ALGORITHMS) continue;
          for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
            offset += snprintf(line+offset, std::max(0, lineLen-offset), "%8.1f/%6.1f |", comm->latencies[c][a][p], comm->bandwidths[c][a][p]);
          }
        }
        INFO(NCCL_TUNING, "%s", line);
      }
    }
  }
 
  // Set per-thread amount of work before we increase nThreads and nChannels
  //线程阈值配置,设置不同算法和协议下的线程阈值
  for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
    comm->threadThresholds[a][NCCL_PROTO_LL] = NCCL_LL_THREAD_THRESHOLD;
    comm->threadThresholds[a][NCCL_PROTO_LL128] = NCCL_LL128_THREAD_THRESHOLD;
    comm->threadThresholds[a][NCCL_PROTO_SIMPLE] = NCCL_SIMPLE_THREAD_THRESHOLD;
  }
  comm->threadThresholds[NCCL_ALGO_RING][NCCL_PROTO_LL] *= nRanks;
  comm->threadThresholds[NCCL_ALGO_COLLNET_DIRECT][NCCL_PROTO_SIMPLE] = 512;
  comm->threadThresholds[NCCL_ALGO_COLLNET_CHAIN][NCCL_PROTO_SIMPLE] = 512;

  // Override defaults with user env 用户自定义覆盖
  const char* str = ncclGetEnv("NCCL_THREAD_THRESHOLDS");
  if (str) {
    INFO(NCCL_ENV, "NCCL_THREAD_THRESHOLDS set by environment to %s", str);
    ssize_t t[2][NCCL_NUM_PROTOCOLS] = {{ -2, -2, -2 }, { -2, -2, -2 }};
    sscanf(str, "%ld %ld %ld %ld %ld %ld", t[0], t[0]+1, t[0]+2, t[1], t[1]+1, t[1]+2);
    for (int a=0; a<2; a++) {
      for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
        if (t[a][p] >= 0) comm->threadThresholds[a][p] = t[a][p];
      }
    }
  }

  INFO(NCCL_INIT, "threadThresholds %ld/%ld/%ld | %ld/%ld/%ld | %ld | %ld",
      comm->threadThresholds[NCCL_ALGO_TREE][NCCL_PROTO_LL],
      comm->threadThresholds[NCCL_ALGO_TREE][NCCL_PROTO_LL128],
      comm->threadThresholds[NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE],
      comm->threadThresholds[NCCL_ALGO_RING][NCCL_PROTO_LL],
      comm->threadThresholds[NCCL_ALGO_RING][NCCL_PROTO_LL128],
      comm->threadThresholds[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE],
      comm->threadThresholds[NCCL_ALGO_COLLNET_DIRECT][NCCL_PROTO_SIMPLE],
      comm->threadThresholds[NCCL_ALGO_COLLNET_CHAIN][NCCL_PROTO_SIMPLE]);
  return ncclSuccess;
}

// Trees are not perfectly sticking to the model for medium sizes. Applying a static correction
// factor is not ideal but works quite well. Powers of two, 64 B to 256MB.
static float treeCorrectionFactor[NCCL_NUM_PROTOCOLS][23] = {
  { 1.0, 1.0, 1.0, 1.0,  .9,  .8,  .7,  .7,  .7,  .7,  .6,  .5,  .4,  .4,  .5,  .6,  .7,  .8,  .9, 1.0, 1.0, 1.0, 1.0 },
  { 1.0, 1.0, 1.0, 1.0, 1.0,  .9,  .8,  .8,  .8,  .7,  .6,  .6,  .6,  .6,  .6,  .6,  .8,  .9,  .9,  .9,  .9, 1.0, 1.0 },
  {  .9,  .9,  .9,  .9,  .9,  .9,  .9,  .8,  .7,  .6,  .6,  .5,  .5,  .5,  .5,  .6,  .7,  .8,  .7,  .7,  .8,  .9,  .9 }
};

ncclResult_t ncclTopoGetAlgoTime(struct ncclComm* comm, int coll, int algorithm, int protocol, size_t nBytes, int numPipeOps, float* time) {
  float bw = comm->bandwidths[coll][algorithm][protocol];
  float lat = comm->latencies[coll][algorithm][protocol];

  if (bw == 0) {
    *time = -1.0; return ncclSuccess;
  }
  int logSize = log2i(nBytes>>6);
  if (algorithm == NCCL_ALGO_TREE && coll == ncclFuncAllReduce && logSize >= 0 && logSize < 23) bw *= treeCorrectionFactor[protocol][logSize];
  if (algorithm == NCCL_ALGO_RING && protocol == NCCL_PROTO_SIMPLE && comm->nNodes > 1
      && coll == ncclFuncAllReduce && nBytes/(comm->nChannels*comm->nRanks) >= 64) {
    lat *= comm->minCompCap < 80 ? 1.9 : 1.4; // Plateau effect of ring
  }
  // Tree pipelining saves latency in aggregation cases
  int latCount = algorithm == NCCL_ALGO_RING ? numPipeOps : DIVUP(numPipeOps, NCCL_MAX_DEV_WORK_BATCH_COLLS);
  *time = lat * latCount + nBytes / (1000 * bw);
  return ncclSuccess;
}
