/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "core.h"
#include "graph.h"
#include "topo.h"
#include "transport.h"
#include "xml.h"
#include <math.h>

NCCL_PARAM(CrossNic, "CROSS_NIC", 2);

// Initialize system->maxBw. This is the per-channel (i.e. per-SM)
// max bw. 
// 获取gpu到指定类型的node的所有路径中的最大带宽。
static float getMaxBw(struct ncclTopoSystem* system, struct ncclTopoNode* gpu, int type) {
  float maxBw = 0.0;
  for (int i=0; i<system->nodes[type].count; i++) {
    struct ncclTopoLinkList* path = gpu->paths[type]+i;
    float bw = path->bw;
    if (path->count == 0) continue;
    maxBw = std::max(maxBw, bw);
  }
  return maxBw;
}
//返回单个gpu的总带宽，即PCI带宽或NVLink带宽的较大值。
static float getTotalBw(struct ncclTopoSystem* system, struct ncclTopoNode* gpu) {
  float nvlinkBw = 0.0, pciBw = 0.0;
  for (int l=0; l<gpu->nlinks; l++) {
    struct ncclTopoLink* link = gpu->links+l;
    if (link->type == LINK_NVL) nvlinkBw += link->bw;
    if (link->type == LINK_PCI) pciBw = link->bw;
  }
  return std::max(pciBw, nvlinkBw);
}
//初始化 NCCL 拓扑系统的带宽信息，为后续通信模式选择和调度提供基础数据。
ncclResult_t ncclTopoSearchInit(struct ncclTopoSystem* system) {
  system->maxBw = 0.0;//gpu到gpu或者网络结点路径中的最大带宽,注意，路径的带宽是路径上所有带宽的最小值。
  system->totalBw = 0.0;//gpu的最大带宽，（nvlink带宽和pcie带宽的最大值）
  int inter = system->nodes[NET].count;
  //如果没有网络节点，并且只有一张GPU，说明是单卡、无网络的极简场景。
  if (inter == 0 && system->nodes[GPU].count == 1) {
    system->maxBw = LOC_BW;
    system->totalBw = LOC_BW;
    return ncclSuccess;
  }
  //遍历所有GPU节点，分别计算每张GPU的最大带宽和总带宽。
  for (int g=0; g<system->nodes[GPU].count; g++) {
    struct ncclTopoNode* gpu = system->nodes[GPU].nodes+g;
    //如果有网络节点，则计算GPU到网络的最大带宽，否则计算GPU到其他GPU的最大带宽。节点内 GPU 之间的 NVLink/PCIe 带宽通常远高于网卡带宽
    system->maxBw = std::max(system->maxBw, getMaxBw(system, gpu, inter ? NET : GPU));
    system->totalBw = std::max(system->totalBw, getTotalBw(system, gpu));//这个是考虑每张GPU的PCI带宽和NVLink带宽的较大值。
    //而maxBw是路径中的最大带宽（与单个节点不同）
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoComputeCommCPU(struct ncclComm* comm) {
  // We assume there is at least one CPU and that the CPUs have the same
  // architecture and vendor. 我们假设至少存在一个CPU，并且所有CPU具有相同的架构和厂商
  const struct ncclTopoNodeSet* cpus = &comm->topo->nodes[CPU];
  comm->cpuArch = cpus->nodes[0].cpu.arch;
  comm->cpuVendor = cpus->nodes[0].cpu.vendor;
  return ncclSuccess;
}
//找到node2到node1的路径
static ncclResult_t findRevLink(struct ncclTopoNode* node1, struct ncclTopoNode* node2, int type, struct ncclTopoLink** revLink) {
  for (int l=0; l<node2->nlinks; l++) {
    struct ncclTopoLink* link = node2->links+l;
    if (link->remNode == node1 && link->type == type) {
      *revLink = link;
      return ncclSuccess;
    }
  }
  WARN("Could not find rev link for %d/%ld -> %d/%ld", node1->type, node1->id, node2->type, node2->id);
  return ncclInternalError;
}

// This is unfortunately needed since manipulating floats often results in rounding errors.
//带宽（ bw ）是用 float 类型表示的。每次分配或释放带宽时，都会做 a = a - b 这样的操.由于浮点数本身的精度有限，连续多次加减后， a 可能会出现微小的误差（比如本来应该是 0，结果变成了 1e-7 或 -1e-6 这种极小的非零值
//- 先做 a-b ，然后乘以 1000，再用 roundf 四舍五入，最后再除以 1000。
//- 这样可以把结果保留到小数点后三位（即精度为 0.001），把极小的误差“归零”。例如：如果结果是 1e-7，经过这个宏处理后就会变成 0。
#define SUB_ROUND(a, b) (a = roundf((a-b)*1000)/1000)
//沿着给定的路径（path），尝试在每一跳上分配（或释放）带宽，并判断路径上是否所有链路都能满足所需带宽。如果中途发现带宽不足，则返回已走过的步数。
static ncclResult_t followPath(struct ncclTopoLinkList* path, struct ncclTopoNode* start, int maxSteps, float bw, int* steps) {
  float pciBw = bw;
  for (int step=0; step<path->count; step++) {
    struct ncclTopoNode* node = path->list[step]->remNode;
    if (node->type == CPU) { //检查是否经过 Intel x86 架构的 CPU，并且路径类型为 PHB（PCI Host Bridge），起点是 GPU。
      //如果是这种情况，说明 GPU 之间通过 Intel CPU 互联，P2P 效率较低，需要对带宽做折算
      // Account for P2P inefficiency through Intel CPU RC
      if (path->type == PATH_PHB && start->type == GPU &&
          node->cpu.arch == NCCL_TOPO_CPU_ARCH_X86 &&
          node->cpu.vendor == NCCL_TOPO_CPU_VENDOR_INTEL) {
        pciBw = INTEL_P2P_OVERHEAD(bw);//这里增大的意思是：消耗的变多了。也即带宽损耗
      }
    }
  }

  struct ncclTopoNode* node = start;//注意node这里其实是指每个边的起点。
  for (int step=0; step<maxSteps; step++) {//遍历路径的每一跳
    struct ncclTopoLink* link = path->list[step];
    struct ncclTopoLink* revLink = NULL;
    float fwBw = link->type == LINK_PCI ? pciBw : bw;//是本跳需要消耗的带宽（如果是 PCI 链路，用上面可能修正过的带宽）。
    float revBw = 0;//是反向链路需要消耗的带宽（某些特殊情况需要双向都扣带宽）
    //如果目标节点是 GPU 且计算能力小于 80，且该跳的起点不是 GPU，反向链路带宽要扣 fwBw/8 （因为旧卡的 P2P 有特殊限制）。
    if (link->remNode->type == GPU && link->remNode->gpu.cudaCompCap < 80 && start->type != GPU) {
      if (revLink == NULL) NCCLCHECK(findRevLink(node, link->remNode, link->type, &revLink));
      revBw += fwBw/8;
    }
    //如果目标节点是 POWER 架构的 CPU 且链路类型是 NVLink，反向链路带宽要扣 fwBw 。
    if (link->remNode->type == CPU && link->remNode->cpu.arch == NCCL_TOPO_CPU_ARCH_POWER && link->type == LINK_NVL) {
      if (revLink == NULL) NCCLCHECK(findRevLink(node, link->remNode, link->type, &revLink));
      revBw += fwBw;
    }
    // Coverity thinks that revLink could be NULL below.  However, we access it only if revBw is non-0, and the
    // logic of the code is that revBw can become non-0 only if revLink is non-NULL (see the "if" statement right above).
    // coverity[var_deref_op]
    // 如果当前链路的剩余带宽小于所需带宽，或者反向链路的剩余带宽小于所需带宽，说明带宽不足，返回已走步数。
    if (link->bw < fwBw || (revBw && revLink->bw < revBw)) { *steps = step; return ncclSuccess; }
    SUB_ROUND(link->bw, fwBw);//用 SUB_ROUND 宏扣减当前链路和反向链路的带宽，防止浮点误差。
    if (revBw) SUB_ROUND(revLink->bw, revBw);
    node = link->remNode;
  }
  *steps = maxSteps;
  return ncclSuccess;
}

// Try to go from node type1/index1 to no type2/index2. mult indicates whether we are counting the bandwidth (1) or undoing (-1).
// 检查并尝试从一个节点到另一个节点的通信路径是否可用（带宽是否足够），如果可用则占用带宽并返回目标节点，如果不可用则回滚带宽占用，保证后续搜索的正确性和资源一致性
// mult:带宽操作方向，1 表示占用带宽，-1 表示释放带宽（回滚）。
static ncclResult_t ncclTopoFollowPath(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, int type1, int index1, int type2, int index2, float mult, struct ncclTopoNode** node) {
  // First handle easy cases
  *node = system->nodes[type2].nodes+index2;
  if (type1 == -1) return ncclSuccess;//初始情况，此时直接返回节点作为第一个点。
  struct ncclTopoNode* node1 = system->nodes[type1].nodes+index1;
  struct ncclTopoLinkList* path = node1->paths[type2]+index2;//a-b的路径
  struct ncclTopoNode* node2 = system->nodes[type2].nodes+index2;
  struct ncclTopoLinkList* revPath = node2->paths[type1]+index1;//b-a的路径

  if (path == NULL) {
    WARN("No path computed to go from %s/%d to %s/%d", topoNodeTypeStr[type1], index1, topoNodeTypeStr[type2], index2);
    return ncclInternalError;
  }

  // Now check link type
  *node = NULL;
  int intra = (type1 == GPU || type1 == NVS) && (type2 == GPU || type2 == NVS);//判断是否为节点内（如 GPU-GPU 或 NVS-GPU）通信。
  float bw = intra ? graph->bwIntra : graph->bwInter;//根据通信类型选择带宽（节点内或节点间）
  int type = intra ? graph->typeIntra : graph->typeInter;
//如果路径类型不满足要求，直接返回成功（跳过该路径）
  if (mult == 1 && (path->type > type)) return ncclSuccess;
  // 这三个 pattern（BALANCED_TREE、TREE、SPLIT_TREE）都是 NCCL 中的树形通信拓扑模式。在这些模式下，通信路径是有方向性的（即数据流动是有父子关系的），而且对于树来说，反向路径的类型（带宽/链路类型）也很重要。
  // 其它模式（如 RING、NVLS、COLLNET）则不需要对反向路径类型做这种严格限制，因为它们的数据流动方式和拓扑结构不同。
  if (mult == 1 && (graph->pattern == NCCL_TOPO_PATTERN_BALANCED_TREE ||
        graph->pattern == NCCL_TOPO_PATTERN_TREE ||
        graph->pattern == NCCL_TOPO_PATTERN_SPLIT_TREE) &&
      (revPath->type > type)) return ncclSuccess;

  bw *= mult;

  // Check there is enough bandwidth on paths.
  int step = 0;
  //检查路径上是否有足够带宽可用，step 表示实际可用的步数。
  NCCLCHECK(followPath(path, node1, path->count, bw, &step));
  if (step < path->count) goto rewind;//说明带宽不足，跳转到 rewind 回滚

  // Enough bandwidth : return destination node.
  graph->nHops += mult*path->count;//更新总跳数。
  *node = system->nodes[type2].nodes+index2;//返回目标节点。
  return ncclSuccess;

rewind:
  // Not enough bandwidth : rewind and exit. 调用 followPath 用 -bw 回滚已占用带宽，保证状态一致性。
  NCCLCHECK(followPath(path, node1, step, -bw, &step));
  return ncclSuccess;
}
// 这个函数就是找到一个pci连接，然后返回其带宽。可能是为了健壮性，这里考虑了双向带宽的最小值。
static int gpuPciBw(struct ncclTopoNode* gpu) {
  for (int l=0; l<gpu->nlinks; l++) {
    struct ncclTopoLink* gpuLink = gpu->links+l;
    if (gpuLink->type != LINK_PCI) continue;
    struct ncclTopoNode* pci = gpuLink->remNode;
    for (int l=0; l<pci->nlinks; l++) {
      struct ncclTopoLink* pciLink = pci->links+l;
      if (pciLink->remNode != gpu) continue;
      return std::min(gpuLink->bw, pciLink->bw);
    }
  }
  return -1;
}

/* Choose the order in which we try next GPUs. This is critical for the search
   to quickly converge to the best solution even if it eventually times out.
   结构体用于对所有候选 GPU 进行多维度打分
   */
struct ncclGpuScore {
  int g;             // Retain the index// 保留GPU的下标（index），用于后续查找
  int startIndex;    // Least important  起始偏移（在环形遍历中的偏移量），排序时优先级最低
  int intraNhops; // 节点内（如NVLink/PCIe）到目标GPU的跳数，跳数越少越优
  int intraBw;// 节点内到目标GPU的带宽，带宽越大越优
  int interNhops; // 跨节点（如网络）到目标GPU的跳数
  int interPciBw;// 跨节点时，目标GPU的PCI带宽
  int interBw;    // Most important 跨节点到目标GPU的网络带宽，排序时优先级最高
};

static int cmpScore(const void * g1, const void * g2) {
   struct ncclGpuScore *s1 = (struct ncclGpuScore*)g1;
   struct ncclGpuScore *s2 = (struct ncclGpuScore*)g2;
   int d;
   if ((d = (s2->interBw - s1->interBw))) return d;
   if ((d = (s2->interPciBw - s1->interPciBw))) return d;
   if ((d = (s1->interNhops - s2->interNhops))) return d;
   if ((d = (s2->intraBw - s1->intraBw))) return d;
   if ((d = (s1->intraNhops - s2->intraNhops))) return d;
   return s1->startIndex - s2->startIndex;//这个意思是最后考虑偏移量小的
}

static int cmpIntraScores(struct ncclGpuScore* scores, int count) {
  int intraBw = scores[0].intraBw;
  int intraNhops = scores[0].intraNhops;
  for (int i=1; i<count; i++) {
    if (scores[i].intraBw != intraBw || scores[i].intraNhops != intraNhops) return 1;
  }
  return 0;
}

static ncclResult_t getGpuIndex(struct ncclTopoSystem* system, int rank, int* index) {
  for (int g=0; g<system->nodes[GPU].count; g++) {
    if (system->nodes[GPU].nodes[g].gpu.rank == rank) {
      *index = g;
      return ncclSuccess;
    }
  }
  WARN("Could not find gpu rank %d", rank);
  return ncclInternalError;
}

static ncclResult_t getNetIndex(struct ncclTopoSystem* system, int64_t id, int* index) {
  for (int n=0; n<system->nodes[NET].count; n++) {
    if (system->nodes[NET].nodes[n].id == id) {
      *index = n;
      return ncclSuccess;
    }
  }
  WARN("Could not find net id %lx", id);
  return ncclInternalError;
}

static ncclResult_t getNetPaths(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, struct ncclTopoLinkList** netPaths) {
  int64_t netId = graph->inter[graph->nChannels*2];
  int n;
  NCCLCHECK(getNetIndex(system, netId, &n));
  *netPaths=system->nodes[NET].nodes[n].paths[GPU];
  return ncclSuccess;
}
//以当前gpu为起点，找下一个gpu节点。这里找到的都是排序后的所有可以到达的gpu，放在next数组中
ncclResult_t ncclTopoSearchNextGpuSort(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, struct ncclTopoNode* gpu, int* next, int* countPtr, int sortNet) {
  const uint64_t flag = 1ULL<<(graph->nChannels);// 当前通道的唯一标志位，用于判断GPU是否已被使用
  int ngpus = system->nodes[GPU].count;
  struct ncclTopoLinkList* paths = gpu->paths[GPU];
  struct ncclTopoLinkList* netPaths = NULL;
  //获取当前channel对应的netid对应的网络设备，然后根据这个网络设备获取其对GPU类型节点的路径。
  if (sortNet) NCCLCHECK(getNetPaths(system, graph, &netPaths));

  struct ncclGpuScore scores[NCCL_TOPO_MAX_NODES];// 用于存储每个候选GPU的评分
  memset(scores, 0, ngpus*sizeof(struct ncclGpuScore));
  int start = gpu-system->nodes[GPU].nodes;//当前gpu为起点。
  int count = 0;
  for (int i=1; i<ngpus; i++) {
    int g = (start+i)%ngpus;// 环形遍历，防止越界
    if (paths[g].count == 0) continue; // There is no path to that GPU 如果没有到该GPU的路径，跳过
    if (system->nodes[GPU].nodes[g].used & flag) continue; // 如果该GPU已被当前通道使用，跳过
    scores[count].g = g;// 记录GPU下标
    scores[count].startIndex = i;// 记录起始偏移
    scores[count].intraNhops = paths[g].count; // 节点内跳数
    scores[count].intraBw = paths[g].bw;// 节点内带宽
    if (netPaths) {
      scores[count].interNhops = netPaths[g].count;
      scores[count].interPciBw = gpuPciBw(system->nodes[GPU].nodes+g);
      scores[count].interBw = netPaths[g].bw;
    }
    count++;
  }

  // Sort GPUs // 对所有候选GPU按评分排序（优先带宽、跳数等）
  qsort(scores, count, sizeof(struct ncclGpuScore), cmpScore);

  // Check if all have the same intra-node score in which case we go reverse for sortNet = -1
  // 如果所有GPU的节点内评分都一样，且sortNet==-1，则逆序排列（用于某些特殊模式下的对称性）
  if (sortNet == -1 && cmpIntraScores(scores, count) == 0) {
    for (int i=0; i<count; i++) next[i] = scores[count-1-i].g;
  } else {
    for (int i=0; i<count; i++) next[i] = scores[i].g;
  }

  *countPtr = count;
  // 如果存在NVSwitch（NVS）节点，优先选择物理相邻的GPU作为通信对象
  if (system->nodes[NVS].count) {
    // NVSwitches prefer when we talk to a limited set of peers. Try to use neighbors first.
    // NVSwitch更倾向于与邻居通信，优先尝试物理相邻的GPU
    int index = gpu-system->nodes[GPU].nodes;// 当前GPU下标
    int i;
    int prevGpu = (index-1+ngpus)%ngpus;// 上一个GPU（环形）
    int nextGpu = (index+1)%ngpus;// 下一个GPU（环形）
    int firstGpus[2];
    int firstGpuCount = 0;
    if (graph->pattern == NCCL_TOPO_PATTERN_RING) {// RING模式下，优先左右邻居
      firstGpus[0] = nextGpu; firstGpus[1] = prevGpu; firstGpuCount = 2;
    } else if (graph->pattern == NCCL_TOPO_PATTERN_SPLIT_TREE ||
        graph->pattern == NCCL_TOPO_PATTERN_BALANCED_TREE) {
      firstGpus[0] = prevGpu; firstGpus[1] = nextGpu; firstGpuCount = 2;// 树形模式下，顺序相反 树形通信中，数据需要先从父节点接收或发送到父节点，所以先选择上一个节点
    } else {
      firstGpus[0] = nextGpu; firstGpuCount = 1; // 其它模式只选下一个
    }
    if (nextGpu == prevGpu && firstGpuCount == 2) firstGpuCount = 1;
    int firstGpuRealCount = 0;// 只有两个GPU时避免重复
    for (int g=0; g<firstGpuCount; g++) {
      for (i=0; i<count && next[i] != firstGpus[g]; i++);// 查找邻居GPU是否在候选列表
      if (i<count) {
        for (; i>0; i--) next[i] = next[i-1];// 将邻居GPU移到最前面
        next[0] = firstGpus[g];
        firstGpuRealCount++;
      }
    }
    *countPtr = firstGpuRealCount;// 只返回邻居GPU数量
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoSearchRec(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, struct ncclTopoGraph* saveGraph, int* time);

// Try to keep all searchs within one second 定义了 NCCL 拓扑搜索相关的超时时间和强制顺序常量
//：全局搜索超时时间，单位是“步数”或“尝试次数”，一般对应约1秒的搜索时间
#define NCCL_SEARCH_GLOBAL_TIMEOUT (1ULL<<19) 
//普通搜索的单次超时时间
#define NCCL_SEARCH_TIMEOUT (1<<14)
//树形拓扑搜索的单次超时时间，和普通搜索一样。
#define NCCL_SEARCH_TIMEOUT_TREE (1<<14)
//用于“相同通道”搜索的超时时间，较短
#define NCCL_SEARCH_TIMEOUT_SAMECHANNELS (1<<8)
//用于控制 GPU 搜索顺序的强制模式，分别代表 PCI 顺序和重放顺序。
#define FORCED_ORDER_PCI 1
#define FORCED_ORDER_REPLAY 2
//得到上一轮channel的第step+1个gpu rank。step为-1，则这里就是上一个channel的首个gpu rank
ncclResult_t ncclTopoReplayGetGpu(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, int step, int* g) {
  *g = -1;
  if (graph->nChannels == 0) return ncclInternalError;
  int ngpus = system->nodes[GPU].count;
  int nextRank = graph->intra[(graph->nChannels-1)*ngpus+step+1];//step为-1，则这里就是上一个channel的第一个gpu rank
  for (int i=0; i<ngpus; i++) if (system->nodes[GPU].nodes[i].gpu.rank == nextRank) {
    *g = i;
    return ncclSuccess;
  }
  if (*g == -1) return ncclInternalError;
  return ncclSuccess;
}

ncclResult_t ncclTopoSearchRecGpu(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, struct ncclTopoGraph* saveGraph, struct ncclTopoNode* gpu, int step, int backToNet, int backToFirstRank, int forcedOrder, int *time);
//尝试以指定的点到指定的GPU 构建通信路径，并通过 used 标志和带宽回滚机制保证搜索的正确性和高效性。
// g是gpu
//ncclTopoSearchTryGpu 负责选出来下一个gpu点，同时看选择出来的下一个点能不能到达
ncclResult_t ncclTopoSearchTryGpu(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, struct ncclTopoGraph* saveGraph, int step, int backToNet, int backToFirstRank, int forcedOrder, int *time, int type, int index, int g) {
  const uint64_t flag = 1ULL<<(graph->nChannels); // 为当前通道生成一个唯一的标志位
  struct ncclTopoNode* gpu;
  // 跟随拓扑路径，从type/index（例如net）到GPU/g，正向计数，找到目标GPU节点，并且更新路径上的边的带宽（减去）
  NCCLCHECK(ncclTopoFollowPath(system, graph, type, index, GPU, g, 1, &gpu));
  if (gpu) {// 如果路径存在且找到GPU
    gpu->used ^= flag;// 用异或操作将该GPU在当前通道的used标志置为已用（或恢复为未用，便于回溯）
    // 递归搜索下一个GPU，尝试构建完整通道
    NCCLCHECK(ncclTopoSearchRecGpu(system, graph, saveGraph, gpu, step, backToNet, backToFirstRank, forcedOrder, time));
    gpu->used ^= flag;// 回溯时恢复used标志，保证搜索树的正确性
    // 路径带宽回滚（撤销本次尝试对带宽的占用）
    NCCLCHECK(ncclTopoFollowPath(system, graph, type, index, GPU, g, -1, &gpu));
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoSearchTryCollnetDirect(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, struct ncclTopoGraph* saveGraph, int g, int ngpus, int *time) {
  int fwdg = 0;
  int bwdg = 0;
  struct ncclTopoNode* gpu = NULL;
  float mul = 1.0 / (float)(system->nodes[GPU].count - 1);
  do {
    NCCLCHECK(ncclTopoFollowPath(system, graph, GPU, g, GPU, fwdg, mul, &gpu));
  } while (gpu && ++fwdg < system->nodes[GPU].count);

  if (gpu != NULL) {
    do {
      NCCLCHECK(ncclTopoFollowPath(system, graph, GPU, bwdg, GPU, g, mul, &gpu));
    } while (gpu && ++bwdg < system->nodes[GPU].count);
    if (gpu != NULL) {
      // Both directions worked. Now we already have head, so pop the all other intra ranks.
      int step = 1;
      for (int index = 0; index < ngpus; ++index) {
        if (index != g) {
          graph->intra[graph->nChannels * ngpus + step] = system->nodes[GPU].nodes[index].gpu.rank;
          step++;
        }
      }
      NCCLCHECK(ncclTopoSearchRecGpu(system, graph, saveGraph, NULL, ngpus, -1, -1, 0, time));
    }
    while (bwdg) {
      bwdg--;
      NCCLCHECK(ncclTopoFollowPath(system, graph, GPU, bwdg, GPU, g, -mul, &gpu));
    }
  }
  while (fwdg) {
    fwdg--;
    NCCLCHECK(ncclTopoFollowPath(system, graph, GPU, g, GPU, fwdg, -mul, &gpu));
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoSearchTryNvls(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, struct ncclTopoGraph* saveGraph, int g, int ngpus, int *time) {
  struct ncclTopoNode* nvs;
  struct ncclTopoNode* gpu;
  int d0=0; // See if there is enough bandwidth for NVS->GPU traffic
  do {
    NCCLCHECK(ncclTopoFollowPath(system, graph, NVS, 0, GPU, d0, d0 == g ? 2 : 1, &gpu));
    d0++;
  } while (gpu && d0 < system->nodes[GPU].count);
  if (gpu == NULL) {
    d0--;
  } else {
    int d1=0; // See if there is enough bandwidth for GPU->NVS traffic
    do {
      NCCLCHECK(ncclTopoFollowPath(system, graph, GPU, d1, NVS, 0, d1 == g ? 2 : 1, &nvs));
      d1++;
    } while (nvs && d1 < system->nodes[GPU].count);
    if (nvs == NULL) {
      d1--;
    } else { // Both directions worked. Move on to the next path.
      NCCLCHECK(ncclTopoSearchRecGpu(system, graph, saveGraph, NULL, ngpus, -1, -1, 0, time));
    }
    while (d1) {
      d1--;
      NCCLCHECK(ncclTopoFollowPath(system, graph, GPU, d1, NVS, 0, d1 == g ? -2 : -1, &nvs));
    }
  }
  while (d0) {
    d0--;
    NCCLCHECK(ncclTopoFollowPath(system, graph, NVS, 0, GPU, d0, d0 == g ? -2 : -1, &gpu));
  }
  return ncclSuccess;
}
//// 比较当前解和历史最优解。
ncclResult_t ncclTopoCompareGraphs(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, struct ncclTopoGraph* refGraph, int* copy) {
  // 1. Try to get the same nChannels between Rings and Trees 如果当前解的通道数（nChannels）小于最小要求（minChannels），直接返回，不做比较。
  if (graph->nChannels < graph->minChannels) return ncclSuccess;

  // 如果当前是 NVLS 模式（NVIDIA NVLink Switch），则优先比较通道数，通道数越多越好（但不能超过GPU总数）。
  //如果通道数相同，则比较通道间带宽（nChannels * bwInter），带宽越大越好。 满足条件则设置 *copy = 1 ，表示当前解更优
  if (graph->pattern == NCCL_TOPO_PATTERN_NVLS) { // NVLS channels correspond to GPUs pulling from NVLS. So the more the better.
    if (graph->nChannels > refGraph->nChannels && graph->nChannels <= system->nodes[GPU].count) *copy = 1;
    if (graph->nChannels*graph->bwInter > refGraph->nChannels*refGraph->bwInter) *copy = 1;
    return ncclSuccess;
  }
  //对于非NVLS模式，优先比较“通道数 × 节点内带宽”（nChannels * bwIntra），带宽越大越好。因为说明可以利用更多的带宽
  // 2. Try to get better bandwidth
  if (graph->nChannels*graph->bwIntra > refGraph->nChannels*refGraph->bwIntra) {
    *copy = 1;
    return ncclSuccess;
  }
  //如果当前解更差，则直接返回。
  if (graph->nChannels*graph->bwIntra < refGraph->nChannels*refGraph->bwIntra) return ncclSuccess;
  //如果带宽相同，则比较跳数（nHops），跳数越少越好（即路径更短，延迟更低）。
  //只有在通信模式（pattern）和跨网卡设置（crossNic）一致时才比较跳数。
  // 3. Less hops
  if (graph->pattern == refGraph->pattern && graph->crossNic == refGraph->crossNic && graph->nHops < refGraph->nHops) *copy = 1;
  return ncclSuccess;
}

// Build a sorted list of the NETs to try.
//
// "gpu" can be set to -1 to build a list suitable for all GPUs (search start) or to a given gpu
//  index when trying to get back to the NIC.
//
// The list is built the following way:
// 1. Select NETs starting with those close to GPU(s), based on paths[n].type.
// 2. add other NETs satisfying typeInter but not already in the list.
// 先优先收集每个GPU直连的NET（本地优先），避免重复。再补充其它满足路径类型要求的NET（如跨NUMA、跨PCI等），同样避免重复。
// 这样得到的NET列表，既保证了优先本地性，又能覆盖所有可用的网络路径，为后续通信拓扑搜索提供候选NIC集合。 这里nets存的是index
ncclResult_t ncclTopoSelectNets(struct ncclTopoSystem* system, int typeInter, int gpu, int* nets, int* netCountRet) {
  ncclResult_t ret = ncclSuccess;
  int netCount = 0;
  int localNetCount;
  int* localNets;
  NCCLCHECK(ncclCalloc(&localNets, MAXCHANNELS));// 分配临时数组存放本地NET索引
  // gpu=-1表示默认搜索全部的，否则只搜索指定的gpu
  // First add the preferred NICs // 首先添加优先的NIC（即每个GPU可达的NIC，而且还是最优带宽的路径）
  for (int g=0; g<system->nodes[GPU].count; g++) {
    if (gpu != -1 && gpu != g) continue;  // 如果指定了gpu，只处理该gpu
    localNetCount = 0;
    struct ncclTopoNode* gpu = system->nodes[GPU].nodes+g;
    for (int c = 0; c<MAXCHANNELS; c++) {
      int64_t netId;
      // 获取本地直连NET（最优的那个）的id
      NCCLCHECKGOTO(ncclTopoGetLocalNet(system, gpu->gpu.rank, c, &netId, NULL), ret, fail);
      //获取对应的索引
      NCCLCHECKGOTO(ncclTopoIdToIndex(system, NET, netId, localNets+localNetCount), ret, fail);
      if (localNetCount > 0 && localNets[localNetCount] == localNets[0]) break;// 如果遇到重复，停止。因为channelId其实就是在自增的。然后取模会出现循环
      localNetCount++;
    }
    // Append NICs to list // 把本地NET加入总列表，避免重复
    for (int i=0; i<localNetCount; i++) {
      int n = localNets[i];
      int found = 0;
      while (nets[found] != n && found<netCount) found++;// 检查当前的net是否已存在
      if (found == netCount) nets[netCount++] = n;// 不存在则加入，否则就不加入
    }
  }

  // Then add others satisfying typeInter
  // 然后补充其它满足typeInter的NET（如跨NUMA、跨PCI等），但不能重复
  for (int t=0; t <= typeInter; t++) {
    for (int g=0; g<system->nodes[GPU].count; g++) {
      if (gpu != -1 && gpu != g) continue;
      localNetCount = 0;
      struct ncclTopoNode* gpu = system->nodes[GPU].nodes+g;
      struct ncclTopoLinkList* paths = gpu->paths[NET];
      for (int n=0; n<system->nodes[NET].count && n<MAXCHANNELS; n++) {
        if (paths[n].type == t) localNets[localNetCount++] = n;// 满足类型的NET加入本地列表
      }
      // Append NICs to list  // 把本地NET加入总列表，避免重复 这里也是对于每个net都看看有没有重复的
      for (int i=0; i<localNetCount; i++) {
        int n = localNets[i];
        int found = 0;
        while (nets[found] != n && found<netCount) found++;
        if (found == netCount) nets[netCount++] = n;
      }
    }
  }

  *netCountRet = netCount;
exit:
  free(localNets);
  return ret;
fail:
  goto exit;
}
//递归地为每个 GPU 动态分配通信路径、带宽，并支持多种通信模式和复杂场景。 递归+回溯
//根据指定的gpu开始，递归地搜索下一个gpu
//用来找下一个GPU，如果一个channel结束，它也会调用ncclTopoSearchRec搜寻新的channel
ncclResult_t ncclTopoSearchRecGpu(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, struct ncclTopoGraph* saveGraph, struct ncclTopoNode* gpu, int step, int backToNet, int backToFirstRank, int forcedOrder, int *time) {
  ncclResult_t ret = ncclSuccess;
  if ((*time) <= 0) return ncclSuccess;//超时，这里其实可以理解为限制递归深度。
  (*time)--;

  int ngpus = system->nodes[GPU].count;
  if (step == ngpus) {// 如果已经遍历完所有GPU，说明本通道构建完成
    // Determine whether we found a better solution or not
    int copy = 0;
    graph->nChannels++;
    // 比较当前解和历史最优解。
    NCCLCHECK(ncclTopoCompareGraphs(system, graph, saveGraph, &copy));
    if (copy) {
      memcpy(saveGraph, graph, sizeof(struct ncclTopoGraph));
      if (graph->nChannels == graph->maxChannels) *time = -1;// 达到最大通道数，提前终止搜索
    }
    // 递归尝试构建下一个通道
    if (graph->nChannels < graph->maxChannels) {
      NCCLCHECK(ncclTopoSearchRec(system, graph, saveGraph, time));
    }
    graph->nChannels--;// 回溯，恢复通道数 注意，前面ncclTopoCompareGraphs后，如果更优，则会吧graph->nChannels++;的结果保存到saveGraph中，等于更新了channel数量。
    //所以这里要把graph->nChannels--;也只是回退临时变量。
    return ncclSuccess;
  }
  graph->intra[graph->nChannels*ngpus+step] = gpu->gpu.rank;//记录一下当前gpu的rank
  int g = gpu - system->nodes[GPU].nodes;// 计算当前GPU在nodes数组中的下标
  int* nets = NULL;
  if (step == backToNet) {// 如果需要回到网络节点（如跨节点通信）backToNet的值: ring：system->nodes[GPU].count-1，split tree：1. 其他都是0
    // first get back to NIC
    if (system->nodes[NET].count) {
      int startNetIndex;
      NCCLCHECK(getNetIndex(system, graph->inter[graph->nChannels*2], &startNetIndex));
      struct ncclTopoNode* startNet = system->nodes[NET].nodes+startNetIndex;
      int netCount;
      NCCLCHECK(ncclCalloc(&nets, system->nodes[NET].count));
      NCCLCHECKGOTO(ncclTopoSelectNets(system, graph->typeInter, g, nets, &netCount), ret, fail);
      for (int i=0; i<netCount; i++) {
        int n = nets[i];
        struct ncclTopoNode* net = system->nodes[NET].nodes+n;
        if (graph->pattern == NCCL_TOPO_PATTERN_TREE && net->id != startNet->id) continue; // Trees are symmetric
        if (graph->pattern == NCCL_TOPO_PATTERN_RING && graph->crossNic == 2) {
          if (graph->nChannels & 1 && net->id != graph->inter[(graph->nChannels-1)*2]) continue;
        } else {
          if (graph->crossNic == 0 && (net->net.asic != startNet->net.asic || net->net.port != startNet->net.port)) continue;
        }

        // Balanced Tree : count half of the bandwidth on first two GPUs
        int nextBackToNet = -1;
        float bwInterSave = graph->bwInter;
        if (graph->pattern == NCCL_TOPO_PATTERN_BALANCED_TREE) {
          // Count half of the bandwidth on each of the first two GPUs
          if (step == 0) nextBackToNet = 1;
          else if (net->id != graph->inter[graph->nChannels*2+1]) continue;
          graph->bwInter /= 2;
        }

        NCCLCHECKGOTO(ncclTopoFollowPath(system, graph, GPU, g, NET, n, 1, &net), ret, fail);
        graph->bwInter = bwInterSave;
        if (net) {
          graph->inter[graph->nChannels*2+1] = net->id;
          NCCLCHECKGOTO(ncclTopoSearchRecGpu(system, graph, saveGraph, gpu, step, nextBackToNet, backToFirstRank, forcedOrder, time), ret, fail);

          if (graph->pattern == NCCL_TOPO_PATTERN_BALANCED_TREE) graph->bwInter /= 2;
          NCCLCHECKGOTO(ncclTopoFollowPath(system, graph, GPU, g, NET, n, -1, &net), ret, fail);
          graph->bwInter = bwInterSave;
        }
      }
    }
  } else if (graph->pattern == NCCL_TOPO_PATTERN_NVLS) {
    NCCLCHECK(ncclTopoSearchTryNvls(system, graph, saveGraph, g, ngpus, time));
  } else if (graph->pattern == NCCL_TOPO_PATTERN_COLLNET_DIRECT) {
    NCCLCHECK(ncclTopoSearchTryCollnetDirect(system, graph, saveGraph, g, ngpus, time));
  } else if (step < system->nodes[GPU].count-1) {
    // Go to next GPU // 还没遍历完所有GPU，递归下一个GPU
    int next[NCCL_TOPO_MAX_NODES];//next 记录了“下一个要递归访问的 GPU 节点编号”
    int count;
    if (forcedOrder == FORCED_ORDER_PCI) { // Try the PCI order  PCI顺序
      next[0] = step+1;
      count = 1;
    } else if (forcedOrder == FORCED_ORDER_REPLAY) { // Try last channel order  重放顺序
      NCCLCHECK(ncclTopoReplayGetGpu(system, graph, step, next));
      count = 1;
    } else { // Normal search
      NCCLCHECK(ncclTopoSearchNextGpuSort(system, graph, gpu, next, &count, backToNet == -1 ? 0 : backToNet == step+1 ? 1 : -1 ));
    }
    //对每个候选的下一个 GPU，递归调用 ncclTopoSearchTryGpu ，以当前gpu g作为起点，尝试以next[i]作为下一个点继续搜索。
    for (int i=0; i<count; i++) {

      NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, step+1, backToNet, backToFirstRank, forcedOrder, time, GPU, g, next[i]));
    }
  } else if (step == backToFirstRank) {//找完一个channel了
    // Find first GPU and loop back to it
    int p;
    NCCLCHECK(getGpuIndex(system, graph->intra[graph->nChannels*ngpus], &p));
    struct ncclTopoNode* firstGpu;
    //尝试从当前 GPU 到第一个 GPU 的路径
    NCCLCHECK(ncclTopoFollowPath(system, graph, GPU, g, GPU, p, 1, &firstGpu));
    if (firstGpu) {//如果路径存在，递归继续搜索，这里其实会导致step==ngpus，然后就相当于结束了。此时也就保存在了saveGraph中
      
      NCCLCHECK(ncclTopoSearchRecGpu(system, graph, saveGraph, firstGpu, step+1, backToNet, -1, forcedOrder, time));
      //恢复带宽占用。因为ncclTopoSearchRecGpu会把未来所有的channel都搜索出来（递归的）搜索完后就要恢复这些修改
      NCCLCHECK(ncclTopoFollowPath(system, graph, GPU, g, GPU, p, -1, &firstGpu));
    }
  } else {
    // Next path 当当前通道的所有 GPU 都已遍历完，且不需要回环时，开始搜索下一个通道，这里step=ngpus了，所以进入下一层就直接构建完成了。然后可能会尝试继续构建
    NCCLCHECK(ncclTopoSearchRecGpu(system, graph, saveGraph, gpu, ngpus, -1, -1, forcedOrder, time));
  }
exit:
  if (nets) free(nets);
  return ret;
fail:
  goto exit;
}
// 该函数负责在多网卡（NET）场景下，递归地为每个通道选择合适的NET节点和起始GPU，并尝试构建高效的通信拓扑。
// 主要流程是：遍历所有可用NET节点，优先选择带宽大、跳数少的本地GPU，支持多种通信模式（Ring、NVLS、CollNet等），并动态管理NET带宽的占用与释放。
ncclResult_t ncclTopoSearchRecNet(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, struct ncclTopoGraph* saveGraph, int backToNet, int backToFirstRank, int* time) {
  ncclResult_t ret = ncclSuccess;
  const int bw = graph->bwInter; // 记录当前通道需要的带宽
  int* nets;// 为所有NET节点分配一个数组
  NCCLCHECK(ncclCalloc(&nets, system->nodes[NET].count));
  int netCount;
  int graphFound = 0;// 标记是否已经找到可行的图（通道）
  // 选择所有可用的NET节点，结果放入nets数组，数量为netCount
  NCCLCHECKGOTO(ncclTopoSelectNets(system, graph->typeInter, -1, nets, &netCount), ret, fail);
  for (int i=0; i<netCount; i++) {// 遍历所有可用的NET节点
    // NVLS和COLLNET_DIRECT模式下，只要找到一个graph就可以break
    if ((graph->pattern == NCCL_TOPO_PATTERN_NVLS || graph->pattern == NCCL_TOPO_PATTERN_COLLNET_DIRECT) && graphFound) break;
    int n = nets[(graph->nChannels+i)%netCount];// 选择当前要尝试的NET节点索引，最开始graph->nChannels=0
    struct ncclTopoNode* net = system->nodes[NET].nodes+n;// 得到NET节点指针
    if (graph->collNet && net->net.collSupport == 0) continue;// 如果需要collNet支持但当前NET不支持，跳过这个net
    if (net->net.bw < bw) continue; // 如果NET带宽不足，跳过这个net
    /*
      Ring模式下crossNic=2且当前通道为奇数通道时，要求NET节点id和上一通道一致，否则跳过
      graph->crossNic == 2 说明允许跨NIC（网卡）通信，并且是“全跨NIC”模式（2通常代表允许所有跨NIC的情况）。
      graph->inter[(graph->nChannels-1)*2+1] 取的是上一个通道（nChannels-1）所选用的第二个NET的id。
      这里的 graph->inter 是一个记录每个通道所选用的NET（NIC）id的数组。
      这句的意思是： 如果当前是奇数通道，并且当前尝试的NET不是上一个通道的第二个NET，则跳过本次循环 。
      比如 [net0, net1, net0, net1, ...]
    */
    if (graph->pattern == NCCL_TOPO_PATTERN_RING && graph->crossNic == 2
        && (graph->nChannels & 1) && net->id != graph->inter[(graph->nChannels-1)*2+1]) continue;
       

    graph->inter[graph->nChannels*2] = net->id; // 记录当前通道使用的NET节点id
    graph->latencyInter = net->net.latency;// 记录当前NET的延迟
      // 占用NET节点带宽（所有同ASIC同端口的NET都减去本通道带宽）
      /*
      asic 指的是网卡（NIC，Network Interface Card）芯片的型号或唯一标识
        - 在一台服务器上，可能有多块物理网卡（每块网卡有自己的 asic），每块网卡上又可能有多个端口（port）。
        - 代码通过判断 asic 和 port ，确保只对同一块物理网卡的同一个端口进行带宽的扣减（即只影响当前实际用到的物理端口）。
      */
    for (int i=0; i<system->nodes[NET].count; i++) {
      if ((system->nodes[NET].nodes[i].net.asic == net->net.asic) &&
          (system->nodes[NET].nodes[i].net.port == net->net.port)) {
        system->nodes[NET].nodes[i].net.bw -= bw;
      }
    }

    if (graph->pattern == NCCL_TOPO_PATTERN_NVLS || graph->pattern == NCCL_TOPO_PATTERN_COLLNET_DIRECT) {
      // NVLS search only tries to find NIC:GPU combinations to compute the heads.
      // 这两种模式下，NCCL 只需要为每个通道（channel）找到一个“头部”GPU与NIC的组合（即 NIC:GPU），而不是像Ring那样构建完整的环路。
      if (graph->nChannels < netCount) {
        int gpu;
        // 查找与当前NIC连接的GPU编号
        NCCLCHECKGOTO(ncclTopoGetLocalGpu(system, net->id, &gpu), ret, fail);
        if (gpu != -1) {
          int duplicate = 0;
          // check whether there is duplicate head when one GPU connects with multiple NICs
          // 检查该GPU是否已经作为其它通道的head，避免重复
          for (int gc = 0; gc < graph->nChannels; gc++) {
            if (graph->intra[gc * system->nodes[GPU].count] == system->nodes[GPU].nodes[gpu].gpu.rank) {
              duplicate = 1;
              break;
            }
          }
          if (!duplicate) {// 如果不是重复
            // 尝试当前NIC+GPU
            NCCLCHECKGOTO(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, 0, time, NET, n, gpu), ret, fail);
            graphFound = 1;
          }
        }
      }
    } else {
      if (graph->nChannels > 0) {
        // Try to replay the last channel
        int g;
        NCCLCHECKGOTO(ncclTopoReplayGetGpu(system, graph, -1, &g), ret, fail);
        NCCLCHECKGOTO(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, FORCED_ORDER_REPLAY, time, NET, n, g), ret, fail);
      }
      if (graph->nChannels == 0 || graph->sameChannels == 0) {
        if (graph->nChannels == 0 && system->nodes[NVS].count == 0) {
          // Always try the PCI order first to set a reference, but don't count in the timeout nor let it run for long
          int t = 1 << 10;
          NCCLCHECKGOTO(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, FORCED_ORDER_PCI, &t, NET, n, 0), ret, fail);
          if (t == -1) *time = -1;
        }

        // Then try the most local GPUs
        float maxBw = 0;
        int minHops = 0xfffffff;
        struct ncclTopoLinkList* paths = net->paths[GPU];
        for (int g=0; g<system->nodes[GPU].count; g++) {
          if (paths[g].bw > maxBw) {
            maxBw = paths[g].bw;
            minHops = paths[g].count;
          } else if (paths[g].bw == maxBw && paths[g].count > 0 && paths[g].count < minHops) {
            minHops = paths[g].count;
          }
        }
        if (maxBw >= bw) {
          for (int i=0; i<system->nodes[GPU].count; i++) {
            int g = (graph->nChannels+i)%system->nodes[GPU].count;
            if (paths[g].bw == maxBw && paths[g].count == minHops) {
              NCCLCHECKGOTO(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, 0, time, NET, n, g), ret, fail);
            }
          }
        }
      }
    }

    for (int i=0; i<system->nodes[NET].count; i++) {
      if ((system->nodes[NET].nodes[i].net.asic == net->net.asic) &&
          (system->nodes[NET].nodes[i].net.port == net->net.port)) {
        system->nodes[NET].nodes[i].net.bw += bw;
      }
    }
  }
exit:
  free(nets);
  return ret;
fail:
  goto exit;
}

/* Search Patterns
 *
 *     Intra-node
 * Ring            : GPU a -> GPU b -> .. -> GPU x -> GPU a
 * (=Split Tree Loop)
 * Tree            : GPU a -> GPU b -> .. -> GPU x
 * (=Split Tree)
 *
 *     Inter-node
 * Ring            : NET n -> GPU a -> GPU b -> .. -> GPU x -> NET n (or m if crossNic)
 * Tree            : NET n -> GPU a -> GPU b -> .. -> GPU x
 *                              `--> NET n (or m if crossNic)
 * Split Tree      : NET n -> GPU a -> GPU b -> .. -> GPU x
 *                                       `--> NET n (or m if crossNic)
 * Split Tree Loop : NET n -> GPU a -> GPU b -> .. -> GPU x -> GPU a
 *                                       `--> NET n (or m if crossNic)
 */
 //backToNet:在跨节点（多机）通信时，递归搜索到一定阶段后，需要回到哪个网络节点（NET）的索引。1 ：表示回到第1个GPU对应的NET节点（如Split Tree模式）。
 //0 表示回到第0个GPU对应的NET节点（如普通Tree模式）。
 //backToFirstRank ：在节点内通信时，回退到第一个GPU的索引。
 //例子：假设是两机16卡（每个机器8卡），此时backToNet=7（意思是7这个卡开始准备返回网络），backToFirstRank=-1；如果是单机8卡，此时backToNet = -1，backToFirstRank=7
ncclResult_t ncclTopoSearchParams(struct ncclTopoSystem* system, int pattern, int* backToNet, int* backToFirstRank) {
  if (system->nodes[NET].count) {//如果系统有网络节点（即多机通信），则根据通信模式（Ring/Split Tree/Tree）设置 backToNet ， backToFirstRank 设为 -1（不用）。
    if (pattern == NCCL_TOPO_PATTERN_RING) *backToNet = system->nodes[GPU].count-1;//Ring：在递归搜索时，最后一个 GPU（索引为 N-1）完成后， 需要回到最初的 NET n ，而不是回到“最后一个 GPU 所在的网络节点”。
    else if (pattern == NCCL_TOPO_PATTERN_SPLIT_TREE) *backToNet = 1;//Split Tree：回到第1个GPU
    else *backToNet = 0; //其它模式：回到第0个
    *backToFirstRank = -1;//只要系统中存在 NET 节点（即多机通信场景），无论是哪种 pattern， *backToFirstRank 都会被设置为 -1。因为此时应该是回到网络
  } else {//单机情况下，要么回到
    *backToNet = -1;
    if (pattern == NCCL_TOPO_PATTERN_RING) *backToFirstRank = system->nodes[GPU].count-1;
    else *backToFirstRank = -1;
  }
  return ncclSuccess;
}
//这个函数就是搜索处来一个路径 对于单机，首先尝试0-1-2-3...的顺序（PCI）然后重复这个channel。如果允许不同的channel，就尝试所有点作为起始点的情况。
ncclResult_t ncclTopoSearchRec(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, struct ncclTopoGraph* saveGraph, int* time) {
  int backToNet, backToFirstRank;//根据当前通信模式（如 Ring、Tree 等）设置回退参数，为后续递归搜索做准备。
  NCCLCHECK(ncclTopoSearchParams(system, graph->pattern, &backToNet, &backToFirstRank));
  if (system->nodes[NET].count) {
    // Start from NET 从net开始出发进行搜索
    ncclTopoSearchRecNet(system, graph, saveGraph, backToNet, backToFirstRank, time);
  } else {
    // Intra-node only.
    if (graph->pattern == NCCL_TOPO_PATTERN_NVLS) {
      NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, 0, time, -1, -1, graph->nChannels));
      return ncclSuccess;
    } else if (graph->nChannels == 0) {
      // Try PCI order first 强制为pci顺序，就是devid的顺序，从dev0开始,其实就是第一个gpu node开始（index=0）。
      //也就是说，最开始nChannels == 0的时候，默认从第一个gpu开始构建。后续都不需要去搜索其他的gpu，都是按照0,1,2,3...的顺序
      //当搜索完第一个channel后，nChannels就不是0了，就会进入另一个分支。
      NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, FORCED_ORDER_PCI, time, -1, -1, 0));
    } else {
      // Also try to replay previous channel
      int g;
      NCCLCHECK(ncclTopoReplayGetGpu(system, graph, -1, &g));//其实就是重复上个channel的gpu顺序
      NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, FORCED_ORDER_REPLAY, time, -1, -1, g));
    }
    if (graph->sameChannels == 0 || graph->nChannels == 0) {
      // Finally, try all other possibilities unless we are forced to use the same channels 
      //尝试其他的路径，尝试以所有 GPU 作为起点，分别递归搜索所有可能的通信路径组合。
      for (int g=0; g<system->nodes[GPU].count; g++) {
        NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, 0, time, -1, -1, g));
      }
    }
  }
  return ncclSuccess;
}

/************************************/
/* User defined graph from XML file */
/************************************/

struct kvDict kvDictLinkType[] = {
  { "LOC", PATH_LOC },
  { "NVL", PATH_NVL },
  { "NVB", PATH_NVB },
  { "PIX", PATH_PIX },
  { "PXB", PATH_PXB },
  { "PXN", PATH_PXN },
  { "PHB", PATH_PHB },
  { "SYS", PATH_SYS },
  { NULL, 0 }
};

ncclResult_t ncclTopoGetChannelFromXml(struct ncclXmlNode *xmlChannel, int c, struct ncclTopoSystem* system, struct ncclTopoGraph* graph) {
  int ngpus = system->nodes[GPU].count;
  int64_t* inter = graph->inter+2*c;
  int* intra = graph->intra+ngpus*c;
  int n=0, g=0;
  for (int s=0; s<xmlChannel->nSubs; s++) {
    struct ncclXmlNode* sub = xmlChannel->subs[s];
    int64_t dev;
    const char* str;
    NCCLCHECK(xmlGetAttrStr(sub, "dev", &str));
    dev = strtol(str, NULL, 16);
    if (strcmp(sub->name, "net") == 0) {
      inter[n++] = dev;
    } else if (strcmp(sub->name, "gpu") == 0) {
      int rank = -1;
      for (int g=0; g<ngpus; g++) {
        int systemId = NCCL_TOPO_ID_SYSTEM_ID(system->nodes[GPU].nodes[g].id);
        if (NCCL_TOPO_ID(systemId, system->nodes[GPU].nodes[g].gpu.dev) == dev) rank = system->nodes[GPU].nodes[g].gpu.rank;
      }
      if (rank == -1) {
        WARN("XML Import Channel : dev %ld not found.", dev);
        return ncclSystemError;
      }
      intra[g++] = rank;
    }
  }
  return ncclSuccess;
}
ncclResult_t ncclTopoGetGraphFromXmlSub(struct ncclXmlNode *xmlGraph, struct ncclTopoSystem* system, struct ncclTopoGraph* graph, int* nChannels) {
  int id;
  NCCLCHECK(xmlGetAttrInt(xmlGraph, "id", &id));
  if (graph->id != id) return ncclSuccess;

  int crossNic;
  NCCLCHECK(xmlGetAttrInt(xmlGraph, "crossnic", &crossNic));
  if (ncclParamCrossNic() == 0 && crossNic == 1) return ncclSuccess;
  graph->crossNic = crossNic;

  NCCLCHECK(xmlGetAttrInt(xmlGraph, "pattern", &graph->pattern));
  NCCLCHECK(xmlGetAttrInt(xmlGraph, "nchannels", &graph->nChannels));
  NCCLCHECK(xmlGetAttrFloat(xmlGraph, "speedintra", &graph->bwIntra));
  NCCLCHECK(xmlGetAttrFloat(xmlGraph, "speedinter", &graph->bwInter));
  if (xmlGetAttrFloat(xmlGraph, "latencyinter", &graph->latencyInter) != ncclSuccess) graph->latencyInter = 0.0;
  const char* str;
  NCCLCHECK(xmlGetAttr(xmlGraph, "typeintra", &str));
  NCCLCHECK(kvConvertToInt(str, &graph->typeIntra, kvDictLinkType));
  NCCLCHECK(xmlGetAttr(xmlGraph, "typeinter", &str));
  NCCLCHECK(kvConvertToInt(str, &graph->typeInter, kvDictLinkType));
  NCCLCHECK(xmlGetAttrInt(xmlGraph, "samechannels", &graph->sameChannels));
  for (int s=0; s<xmlGraph->nSubs; s++) {
    NCCLCHECK(ncclTopoGetChannelFromXml(xmlGraph->subs[s], s, system, graph));
  }
  *nChannels = xmlGraph->nSubs;
  return ncclSuccess;
}
ncclResult_t ncclTopoGetGraphFromXml(struct ncclXmlNode *xmlGraphs, struct ncclTopoSystem* system, struct ncclTopoGraph* graph, int* nChannels) {
  for (int s=0; s<xmlGraphs->nSubs; s++) {
    NCCLCHECK(ncclTopoGetGraphFromXmlSub(xmlGraphs->subs[s], system, graph, nChannels));
  }
  return ncclSuccess;
}

/* And the reverse : graph->xml */
ncclResult_t ncclTopoGetXmlFromChannel(struct ncclTopoGraph* graph, int c, struct ncclTopoSystem* system, struct ncclXml *xml, struct ncclXmlNode* parent) {
  struct ncclXmlNode* xmlChannel;
  int ngpus = system->nodes[GPU].count;
  int64_t* inter = graph->inter+2*c;
  int* intra = graph->intra+ngpus*c;
  NCCLCHECK(xmlAddNode(xml, parent, "channel", &xmlChannel));
  struct ncclXmlNode* node;
  if (system->nodes[NET].count) {
    NCCLCHECK(xmlAddNode(xml, xmlChannel, "net", &node));
    NCCLCHECK(xmlSetAttrLong(node, "dev", inter[0]));
  }
  for (int g=0; g<ngpus; g++) {
    NCCLCHECK(xmlAddNode(xml, xmlChannel, "gpu", &node));
    int64_t dev = -1;
    for (int i=0; i<ngpus; i++) {
      if (system->nodes[GPU].nodes[i].gpu.rank == intra[g]) {
        int systemId = NCCL_TOPO_ID_SYSTEM_ID(system->nodes[GPU].nodes[i].id);
        dev = NCCL_TOPO_ID(systemId, system->nodes[GPU].nodes[i].gpu.dev);
      }
    }
    if (dev == -1) {
      WARN("XML Export Channel : rank %d not found.", intra[g]);
      return ncclInternalError;
    }
    NCCLCHECK(xmlSetAttrLong(node, "dev", dev));
    if (graph->id == 3) break; // NVLS graphs only use the first GPU
  }
  if (system->nodes[NET].count) {
    NCCLCHECK(xmlAddNode(xml, xmlChannel, "net", &node));
    NCCLCHECK(xmlSetAttrLong(node, "dev", inter[1]));
  }
  return ncclSuccess;
}
ncclResult_t ncclTopoGetXmlFromGraph(struct ncclTopoGraph* graph, struct ncclTopoSystem* system, struct ncclXml *xml, struct ncclXmlNode* parent) {
  struct ncclXmlNode* xmlGraph;
  NCCLCHECK(xmlAddNode(xml, parent, "graph", &xmlGraph));
  NCCLCHECK(xmlSetAttrInt(xmlGraph, "id", graph->id));
  NCCLCHECK(xmlSetAttrInt(xmlGraph, "pattern", graph->pattern));
  NCCLCHECK(xmlSetAttrInt(xmlGraph, "crossnic", graph->crossNic));
  NCCLCHECK(xmlSetAttrInt(xmlGraph, "nchannels", graph->nChannels));
  NCCLCHECK(xmlSetAttrFloat(xmlGraph, "speedintra", graph->bwIntra));
  NCCLCHECK(xmlSetAttrFloat(xmlGraph, "speedinter", graph->bwInter));
  NCCLCHECK(xmlSetAttrFloat(xmlGraph, "latencyinter", graph->latencyInter));
  const char* str;
  NCCLCHECK(kvConvertToStr(graph->typeIntra, &str, kvDictLinkType));
  NCCLCHECK(xmlSetAttr(xmlGraph, "typeintra", str));
  NCCLCHECK(kvConvertToStr(graph->typeInter, &str, kvDictLinkType));
  NCCLCHECK(xmlSetAttr(xmlGraph, "typeinter", str));
  NCCLCHECK(xmlSetAttrInt(xmlGraph, "samechannels", graph->sameChannels));
  for (int c=0; c<graph->nChannels; c++) {
    NCCLCHECK(ncclTopoGetXmlFromChannel(graph, c, system, xml, xmlGraph));
  }
  return ncclSuccess;
}
ncclResult_t ncclTopoGetXmlFromGraphs(int ngraphs, struct ncclTopoGraph** graphs, struct ncclTopoSystem* system, struct ncclXml *xml) {
  xml->maxIndex = 0;
  struct ncclXmlNode* xmlGraphs;
  NCCLCHECK(xmlAddNode(xml, NULL, "graphs", &xmlGraphs));
  NCCLCHECK(xmlSetAttrInt(xmlGraphs, "version", NCCL_GRAPH_XML_VERSION));
  for (int g=0; g<ngraphs; g++) {
    NCCLCHECK(ncclTopoGetXmlFromGraph(graphs[g], system, xml, xmlGraphs));
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoDupChannels(struct ncclTopoGraph* graph, int ccMin, int ngpus) {
  if (graph->nChannels == 0) return ncclSuccess;
  if (graph->pattern == NCCL_TOPO_PATTERN_NVLS) return ncclSuccess;
  if (graph->bwIntra < 25.0) return ncclSuccess;
  if (ccMin > 80 && graph->bwIntra < 50.0 && graph->nChannels > 4) return ncclSuccess;

  int dupChannels = std::min(graph->nChannels*2, graph->maxChannels);
  memcpy(graph->intra+graph->nChannels*ngpus, graph->intra, (dupChannels-graph->nChannels)*ngpus*sizeof(int));
  memcpy(graph->inter+graph->nChannels*2,graph->inter, (dupChannels-graph->nChannels)*2*sizeof(int64_t));
  graph->bwIntra /= DIVUP(dupChannels, graph->nChannels);
  graph->bwInter /= DIVUP(dupChannels, graph->nChannels);
  graph->nChannels = dupChannels;
  return ncclSuccess;
}
//这几行代码定义了 NCCL 在不同 GPU 架构和通信场景下的“带宽候选数组”
//节点内（如同一台机器内 GPU 之间）通信的带宽候选值（单位 GB/s），适用于较老架构。
float speedArrayIntra[] = { 40.0, 30.0, 20.0, 18.0, 15.0, 12.0, 10.0, 9.0, 7.0, 6.0, 5.0, 4.0, 3.0 };
//节点间（如跨机器）通信的带宽候选值
float speedArrayInter[] = { 48.0, 30.0, 28.0, 24.0, 20.0, 18.0, 15.0, 12.0, 10.0, 9.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.4, 1.2, 0.24, 0.12 };
//分别表示每个带宽数组的元素个数，便于后续遍历。
#define NSPEEDSINTRA (sizeof(speedArrayIntra)/sizeof(float)) 
#define NSPEEDSINTER (sizeof(speedArrayInter)/sizeof(float))

float sm90SpeedArrayIntra[] = { 60.0, 50.0, 40.0, 30.0, 24.0, 20.0, 15.0, 12.0, 11.0, 6.0, 3.0 };
float sm90SpeedArrayInter[] = { 48.0, 45.0, 42.0, 40.0, 30.0, 24.0, 22.0, 20.0, 17.5, 15.0, 12.0, 6.0, 3.0, 2.4, 1.2, 0.24, 0.12 };
#define NSPEEDSINTRA_SM90 (sizeof(sm90SpeedArrayIntra)/sizeof(float))
#define NSPEEDSINTER_SM90 (sizeof(sm90SpeedArrayInter)/sizeof(float))

float sm100SpeedArrayIntra[] = { 90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0, 24.0, 20.0, 19.0 };
float sm100SpeedArrayInter[] = { 48.0, 45.0, 42.0, 40.0, 30.0, 24.0, 22.0, 20.0, 17.5, 15.0, 12.0, 6.0, 3.0, 2.4, 1.2, 0.24, 0.12 };
#define NSPEEDSINTRA_SM100 (sizeof(sm100SpeedArrayIntra)/sizeof(float))
#define NSPEEDSINTER_SM100 (sizeof(sm100SpeedArrayInter)/sizeof(float))

//负责根据硬件结构和通信模式，自动生成最优的通信路径和参数
//这里就是实际搜索channel的过程，目标是搜索出来尽可能多，带宽尽可能大的一系列channel，
// 本质就是暴力搜索，先设置一系列的条件搜答案，如果搜不出来则降低条件继续搜。
//其实决定channel的个数因素是path->bw和graph->bwIntra(有网卡时为graph->bwInter)，即一个path的带宽（path->bw）可以容纳几个channel带宽graph->bwIntra。
//此外就是这个主要是针对单节点，因为topo结构就是单节点的。
ncclResult_t ncclTopoCompute(ncclTopoSystem* system, struct ncclTopoGraph* graph) {
  int ngpus = system->nodes[GPU].count;
  //判断是否需要跨网卡（crossNic）跨网卡通信（crossNic） ，就是指在单节点内部，合理分配GPU到不同NIC的通信路径，使得多块网卡都能被充分利用。
  //注意，如果是单机，前面的trim已经把网卡node删掉了。
  //这里inter主要是指gpu到网卡，intra是gpu到gpu
  int crossNic = (system->nodes[NET].count > 1) &&
	 (graph->pattern == NCCL_TOPO_PATTERN_RING ||
          graph->pattern == NCCL_TOPO_PATTERN_BALANCED_TREE ||
          graph->pattern == NCCL_TOPO_PATTERN_SPLIT_TREE) ? ncclParamCrossNic() : 0;
  graph->crossNic = crossNic == 1 ? 1 : 0;
  graph->bwIntra = graph->bwInter = 0;
  graph->latencyInter = 0;
  //GPU 到 NET（NIC）之间的路径最大只允许到系统内部的极限，不会跨主机
  //- intra ：通常指同一个 NUMA 节点内的 GPU 之间通信（比如同一个 CPU 下的 GPU 通过 PCIe/NVLink 互联）。
  //- inter ：指不同 NUMA 节点之间的通信，哪怕还在同一台物理服务器上，但因为要跨 CPU/主板互连，性能大幅下降，所以 NCCL 会把这种路径归为“inter”。
  int minTypeIntra = PATH_LOC, minTypeInter = PATH_PIX;
  int maxTypeIntra = PATH_SYS, maxTypeInter = PATH_SYS;
  if (ngpus > 1) {
    //GPU之间通信路径的最小类型
    NCCLCHECK(ncclTopoGetGpuMinPath(system, GPU, &minTypeIntra));
    //GPU之间通信路径的最大类型
    NCCLCHECK(ncclTopoGetGpuMaxPath(system, GPU, &maxTypeIntra));
  }
  if (system->nodes[NET].count > 0) {
    //获取GPU到NET的最小和最大路径类型
    NCCLCHECK(ncclTopoGetGpuMinPath(system, NET, &minTypeInter));
    NCCLCHECK(ncclTopoGetGpuMaxPath(system, NET, &maxTypeInter));
    maxTypeIntra = maxTypeInter;/* 这里AI的解释如下（但是我觉得有点问题，可能要等全部代码看完后才能懂）：
    但如果系统中有网络节点（NET） ，说明通信模式可能涉及跨节点（多机多卡），这时 NCCL 需要保证节点内和节点间的路径类型选择是一致的，避免出现“节点内允许的路径类型比节点间更宽松”导致的通信不一致或性能问题。
    - 如果 maxTypeInter 是 PATH_SYS，说明跨节点通信最多允许到系统级别的路径（如跨 NUMA、跨主板）。
    - 这时把 maxTypeIntra 也设为 PATH_SYS，意味着节点内通信也不能走比 PATH_SYS 更“远”的路径（比如不能走 PATH_NET，因为那是跨主机的网络路径）。*/
    //实际上这里最大为NET
  }
  graph->typeIntra = minTypeIntra; //注意都是用最小的类型初始化的。
  graph->typeInter = minTypeInter;
  graph->nChannels = 0;
  int trySameChannels = graph->pattern == NCCL_TOPO_PATTERN_NVLS ? 0 : 1;
  graph->sameChannels = trySameChannels;//sameChannel设置为0，允许channel之间不一样 

  int cpuArch, cpuVendor, cpuModel;
  NCCLCHECK(ncclTopoCpuType(system, &cpuArch, &cpuVendor, &cpuModel));
  //优先从环境变量加载通信拓扑
  const char* str = ncclGetEnv("NCCL_GRAPH_FILE");
  if (str) {
    INFO(NCCL_ENV, "NCCL_GRAPH_FILE set by environment to %s", str);
    struct ncclXml* xml;
    NCCLCHECK(xmlAlloc(&xml, NCCL_GRAPH_XML_MAX_NODES));
    NCCLCHECK(ncclTopoGetXmlGraphFromFile(str, xml));
    int nChannels;
    NCCLCHECK(ncclTopoGetGraphFromXml(xml->nodes, system, graph, &nChannels));
    INFO(NCCL_GRAPH, "Search %d : %d channels loaded from XML graph", graph->id, nChannels);
    free(xml);
    if (graph->nChannels > 0) return ncclSuccess;//如果成功加载并且 graph->nChannels > 0 ，说明已经有可用的通信通道，直接返回，不再进行后续的自动搜索。
  }

  int ccMin;//获取系统中所有GPU的最低计算能力
  NCCLCHECK(ncclTopoGetCompCap(system, &ccMin, NULL));

  /*
  下面这些channel的设置要重新思考一下原理，感觉ai给的解释不太对。
  */

    //如果当前通信模式是 NVLS或者最低计算能力小于90（即不是Hopper架构及以上），直接返回，不进行NVLS搜索。
  if (graph->pattern == NCCL_TOPO_PATTERN_NVLS && (system->nodes[NVS].count == 0 || ccMin < 90)) return ncclSuccess;
  // NVLS and COLLNET_DIRECT search must have ngpus heads at most.

  //对于 NVLS 模式，最大通道数不能超过 NCCL_MAX_NVLS_ARITY 和GPU数量的较小值。
  //我猜是因为nvswitch是相当于全互联，那么每个gpu到其他gpu都相当于有连接。
  if (graph->pattern == NCCL_TOPO_PATTERN_NVLS) graph->maxChannels = std::min(NCCL_MAX_NVLS_ARITY, system->nodes[GPU].count);
 //对于 COLLNET_DIRECT 模式，最大通道数不能超过 NCCL_MAX_DIRECT_ARITY+1 和GPU数量的较小值。
 //这样做是为了保证通道数不会超过硬件和算法支持的上限。
  if (graph->pattern == NCCL_TOPO_PATTERN_COLLNET_DIRECT) graph->maxChannels = std::min(NCCL_MAX_DIRECT_ARITY+1, system->nodes[GPU].count);
  //如果只有一张GPU（ngpus == 1），且当前模式不是RING，则强制切换为TREE模式。因为单卡下RING没有意义，TREE更合适。
  if (ngpus == 1) if (graph->pattern != NCCL_TOPO_PATTERN_RING) graph->pattern = NCCL_TOPO_PATTERN_TREE;
//如果没有网络节点（NET），且当前为NVLS模式，说明是单机NVSwitch场景。
  if (system->nodes[NET].count == 0 && graph->pattern == NCCL_TOPO_PATTERN_NVLS) {
    // Force intra-node NVLS algorithm to pull evenly from all GPUs.
    graph->minChannels = graph->maxChannels;//即所有GPU均匀地参与所有通道，保证带宽利用最大化。
  }



  struct ncclTopoGraph tmpGraph;
  memcpy(&tmpGraph, graph, sizeof(struct ncclTopoGraph));
//根据硬件环境和通信模式，选择合适的带宽参数（speedArray[speedIndex]），为后续的通信拓扑搜索做准备
  // First try crossnic, then decrease bw and finally increase bwIntra.
  int nspeeds = 0;
  float* speedArray = NULL;
  if (system->nodes[NET].count == 0) {//，说明是单机通信，选择节点内带宽数组
    nspeeds = ccMin >= 100 ? NSPEEDSINTRA_SM100 : (ccMin >= 90 ? NSPEEDSINTRA_SM90 : NSPEEDSINTRA);
    speedArray = ccMin >= 100 ? sm100SpeedArrayIntra : (ccMin >= 90 ? sm90SpeedArrayIntra : speedArrayIntra);
  } else {
    nspeeds = ccMin >= 100 ? NSPEEDSINTER_SM100 : (ccMin >= 90 ? NSPEEDSINTER_SM90 : NSPEEDSINTER);
    speedArray = ccMin >= 100 ? sm100SpeedArrayInter : (ccMin >= 90 ? sm90SpeedArrayInter : speedArrayInter);
  }
  int pass = 1;
  int speedIndex = 0;
  float maxBw = system->maxBw;//gpu到gpu或者网络结点路径中的最大带宽（整个路径的最小带宽）（也就是单通道的最大带宽）
  float totalBw = system->totalBw;//注意这里的totalBw是单个gpu的最大带宽
  //如果是多卡且不是 ring 模式， totalBw 乘以 ngpus/(ngpus-1) ，这是为了补偿非环形通信模式下的带宽分摊。
  if (ngpus > 1 && graph->pattern != NCCL_TOPO_PATTERN_RING) totalBw *= ngpus*1.0/(ngpus-1);
  //通过 while 循环，找到一个既不超过所有路径的最大带宽、又能满足最小通道数需求的带宽下标 speedIndex 。
  while ((speedArray[speedIndex] > maxBw || speedArray[speedIndex]*graph->minChannels > totalBw) && speedIndex < nspeeds-1) speedIndex++;
  tmpGraph.bwIntra = tmpGraph.bwInter = speedArray[speedIndex];
  int64_t globalTimeout = NCCL_SEARCH_GLOBAL_TIMEOUT;//用于控制整个搜索过程的最大耗时。

  //通过不断调整通信参数（如通道一致性、拓扑模式、链路类型、带宽、跨网卡等），递归尝试各种组合，直到找到最优或可行的通信拓扑方案为止。
search:
  //根据当前是否要求“相同通道”或是树型模式，选择不同的超时时间。
  int time = tmpGraph.sameChannels ? NCCL_SEARCH_TIMEOUT_SAMECHANNELS :
    tmpGraph.pattern == NCCL_TOPO_PATTERN_TREE ? NCCL_SEARCH_TIMEOUT_TREE : NCCL_SEARCH_TIMEOUT;
  tmpGraph.nChannels = 0;//初始化通道数
  globalTimeout -= time;//全局超时递减，防止整体搜索时间过长
  //调用 ncclTopoSearchRec 递归搜索拓扑方案
  NCCLCHECK(ncclTopoSearchRec(system, &tmpGraph, graph, &time));
#if 0
  printf("Id %d Pattern %d, crossNic %d, Bw %g/%g, type %d/%d, channels %d-%d sameChannels %d -> nChannels %dx%g/%g %s\n", tmpGraph.id, tmpGraph.pattern, tmpGraph.crossNic, tmpGraph.bwInter, tmpGraph.bwIntra, tmpGraph.typeInter, tmpGraph.typeIntra, tmpGraph.minChannels, tmpGraph.maxChannels, tmpGraph.sameChannels, graph->nChannels, graph->bwInter, graph->bwIntra, time == 0 ? "TIMEOUT" : time == -1 ? "PERFECT" : "");
  for (int c=0; c<graph->nChannels; c++) {
    printf("%2d : ", c);
    for (int g=0; g<ngpus; g++) {
      printf("%d ", graph->intra[c*ngpus+g]);
    }
    printf("[%lx %lx]", graph->inter[c*2+0], graph->inter[c*2+1]);
    printf("\n");
  }
#endif
  // Optimal solution, stop here
  if (time == -1) goto done;//表示找到了完美解决方案
  if (graph->nChannels*graph->bwInter >= system->totalBw) goto done;//表示已经达到系统总带宽上限，无需继续优化

  if (pass == 1) {
    // First pass, we don't have a solution yet ; try other options 第一阶段的主要目标是找到一个可行解

    // Try having different channels (except when going through AMD CPUs) 
    //首先尝试使用不同的通道配置（除非是通过 AMD CPU 的特殊情况）
    if (tmpGraph.sameChannels == 1 &&
        !(cpuArch == NCCL_TOPO_CPU_ARCH_X86 && cpuVendor == NCCL_TOPO_CPU_VENDOR_AMD && tmpGraph.typeIntra == PATH_SYS)) {
      tmpGraph.sameChannels = 0;
      goto search;
    }
    tmpGraph.sameChannels = trySameChannels;
    /*
    - 如果搜索未超时，将剩余时间加回全局超时
    - 如果已经找到完美解，重置全局超时
    - 如果全局超时已耗尽且已有解决方案，直接结束搜索
    */
    if (time != -1) globalTimeout += time;
    else globalTimeout = NCCL_SEARCH_GLOBAL_TIMEOUT;
    if (globalTimeout < 0 && graph->nChannels) goto done;

    // Try a simpler tree 对于较新的 GPU（计算能力 >= 9.0），尝试从平衡树降级到简单树模式。
    if (ccMin >= 90 && tmpGraph.pattern == NCCL_TOPO_PATTERN_BALANCED_TREE) {
      tmpGraph.pattern = NCCL_TOPO_PATTERN_TREE;
      goto search;
    }
    tmpGraph.pattern = graph->pattern;
    //尝试提高节点内通信类型（如从 NVLink 升级到 PCIe）
    int maxIntra = system->nodes[NET].count > 0 ? tmpGraph.typeInter : maxTypeIntra;
    if (tmpGraph.typeIntra < maxIntra && (graph->nChannels == 0 || tmpGraph.typeIntra < graph->typeIntra)) {
      tmpGraph.typeIntra += 1;
      goto search;
    }
    tmpGraph.typeIntra = minTypeIntra;
    //如果有网络节点，尝试提高节点间通信类型。
    if (system->nodes[NET].count > 0 && tmpGraph.typeInter < maxTypeInter && (graph->nChannels == 0 || tmpGraph.typeInter < graph->typeInter || tmpGraph.typeInter < PATH_PXN)) {
      tmpGraph.typeInter += 1;
      goto search;
    }
    tmpGraph.typeInter = minTypeInter;
    //对于环形或平衡树模式，尝试启用跨网卡通信。
    if (crossNic == 2 && tmpGraph.crossNic == 0
        && (graph->pattern == NCCL_TOPO_PATTERN_RING || graph->pattern == NCCL_TOPO_PATTERN_BALANCED_TREE)) {
      // Try again with crossNic if permitted
      tmpGraph.crossNic = 2;
      goto search;
    }
    tmpGraph.crossNic = crossNic == 1 ? 1 : 0;
    //如果以上调整都不成功，尝试降低带宽要求，直到找到可行解。
    // Decrease bw until we find a solution
    if ((speedIndex < nspeeds-1) && (graph->nChannels == 0 || (speedArray[speedIndex+1]/graph->bwInter > .49))) {
      tmpGraph.bwInter = tmpGraph.bwIntra = speedArray[++speedIndex];
      goto search;
    }
    speedIndex = 0;
    while (speedArray[speedIndex] > maxBw && speedIndex < nspeeds-1) speedIndex++;
    tmpGraph.bwIntra = tmpGraph.bwInter = speedArray[speedIndex];

  }

done:
  // We have a solution. Start from that solution and move to pass 2.
  if (pass == 1) {
    time = -1;
    NCCLCHECK(ncclTopoDupChannels(graph, ccMin, ngpus));//复制通道配置
    memcpy(&tmpGraph, graph, sizeof(tmpGraph));//重置临时图为当前最佳解决方案
    speedIndex = 0;//设置带宽索引
    while (speedArray[speedIndex] > graph->bwInter && speedIndex < nspeeds-1) speedIndex++;
    tmpGraph.bwIntra = tmpGraph.bwInter = speedArray[speedIndex];
    tmpGraph.minChannels = graph->nChannels;//将最小通道数设为当前通道数，防止后续优化过程中通道数减少
    pass = 2;//进入第二阶段
  }

  if (pass == 2) {//目标是在保持通道数不变的情况下，尝试提高带宽
    // See if we can increase bw
    if (time != 0 && speedIndex > 0) {
      if (graph->pattern == NCCL_TOPO_PATTERN_RING) {//对于环形模式，同时提高节点内和节点间带宽
        // increase bw for Ring
        tmpGraph.bwIntra = tmpGraph.bwInter = speedArray[--speedIndex];
        goto search;
      } else if (graph->pattern == NCCL_TOPO_PATTERN_NVLS && tmpGraph.bwInter == graph->bwInter && tmpGraph.bwInter < tmpGraph.bwIntra*2) {
        //对于 NVLS 模式，只提高节点间带宽
        tmpGraph.minChannels = tmpGraph.maxChannels = graph->nChannels;
        tmpGraph.bwInter = speedArray[--speedIndex];
        goto search;
      } else if (tmpGraph.bwIntra == graph->bwIntra && tmpGraph.bwIntra < tmpGraph.bwInter*2) {
        //对于树形模式，只提高节点内带宽
        // increase bwIntra for trees (2 nodes or collnet)
        tmpGraph.bwIntra = speedArray[--speedIndex];
        goto search;
      }
    }
    time = -1;
    memcpy(&tmpGraph, graph, sizeof(tmpGraph));
  }
  //如果所有尝试都失败，提供一个最基本的兜底方案： 设置最低带宽和最基本的通信类型 设置最低带宽和最基本的通信类型 使用单通道配置
  if (graph->nChannels == 0 && graph->collNet == 0 && graph->pattern != NCCL_TOPO_PATTERN_NVLS) {
    WARN("Could not find a path for pattern %d, falling back to simple order", graph->pattern);
    for (int i=0; i<ngpus; i++) graph->intra[i] = system->nodes[GPU].nodes[i].gpu.rank;
    graph->inter[0] = graph->inter[1] = 0;
    graph->bwIntra = graph->bwInter = 0.1;
    graph->typeIntra = graph->typeInter = PATH_SYS;
    graph->nChannels = 1;
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoPrintGraph(struct ncclTopoSystem* system, struct ncclTopoGraph* graph) {
  INFO(NCCL_GRAPH, "Pattern %d, crossNic %d, nChannels %d, bw %f/%f, type %s/%s, sameChannels %d", graph->pattern, graph->crossNic, graph->nChannels, graph->bwIntra, graph->bwInter, topoPathTypeStr[graph->typeIntra], topoPathTypeStr[graph->typeInter], graph->sameChannels);
  int ngpus = system->nodes[GPU].count;

  char line[1024];
  for (int c=0; c<graph->nChannels; c++) {
    sprintf(line, "%2d :", c);
    int offset = strlen(line);
    if (system->nodes[NET].count > 0) {
      sprintf(line+offset, " %s/%lx-%lx", topoNodeTypeStr[NET], NCCL_TOPO_ID_SYSTEM_ID(graph->inter[2*c]), NCCL_TOPO_ID_LOCAL_ID(graph->inter[2*c]));
      offset = strlen(line);
    }
    for (int i=0; i<ngpus; i++) {
      sprintf(line+offset, " %s/%d", topoNodeTypeStr[GPU], graph->intra[ngpus*c+i]);
      offset = strlen(line);
    }
    if (system->nodes[NET].count > 0) {
      sprintf(line+offset, " %s/%lx-%lx", topoNodeTypeStr[NET], NCCL_TOPO_ID_SYSTEM_ID(graph->inter[2*c+1]), NCCL_TOPO_ID_LOCAL_ID(graph->inter[2*c+1]));
      offset = strlen(line);
    }
    INFO(NCCL_GRAPH, "%s", line);
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoDumpGraphs(struct ncclTopoSystem* system, int ngraphs, struct ncclTopoGraph** graphs) {
  ncclResult_t ret = ncclSuccess;
  const char* str = ncclGetEnv("NCCL_GRAPH_DUMP_FILE");
  struct ncclXml* xml = NULL;
  if (str) {
    INFO(NCCL_ENV, "NCCL_GRAPH_DUMP_FILE set by environment to %s", str);
    NCCLCHECK(xmlAlloc(&xml, NCCL_GRAPH_XML_MAX_NODES));
    NCCLCHECKGOTO(ncclTopoGetXmlFromGraphs(ngraphs, graphs, system, xml), ret, fail);
    NCCLCHECKGOTO(ncclTopoDumpXmlToFile(str, xml), ret, fail);
  }
exit:
  if (xml) free(xml);
  return ret;
fail:
  goto exit;
}

#include "comm.h"
// NVLS channels aren't compute channels. Find which NIC corresponds to our rank being the head
ncclResult_t getNvlsNetDev(struct ncclComm* comm, struct ncclTopoGraph* graph, int channelId, int64_t* netId) {
  ncclResult_t ret = ncclSuccess;
  int localRanks = comm->topo->nodes[GPU].count;
  int netNum = 0;
  int64_t net[MAXCHANNELS];

  for (int c = 0; c < graph->nChannels; c++) {
    if (graph->intra[c * localRanks] == comm->rank) {
      net[netNum++] = graph->inter[c * 2];
    }
  }
  if (netNum) {
    *netId = net[channelId % netNum];
  } else {
    ret = ncclInternalError;
    goto fail;
  }

exit:
  return ret;
fail:
  WARN("Could not find NIC for rank %d in NVLS graph", comm->rank);
  goto exit;
}

// 0: don't use PXN for P2P, 1: use PXN if needed, 2: use PXN as much as possible to maximize aggregation
NCCL_PARAM(P2pPxnLevel, "P2P_PXN_LEVEL", 2);

ncclResult_t ncclTopoGetNetDev(struct ncclComm* comm, int rank, struct ncclTopoGraph* graph, int channelId, int peerRank, int64_t* id, int* dev, int* proxyRank) {
  int64_t netId = -1;
  int netDev = -1;
  if (graph) {
    // Honor the net device in the graph
    int channel = channelId%graph->nChannels;
    int ngpus = comm->topo->nodes[GPU].count;
    int index = graph->intra[channel*ngpus] == rank ? 0 : 1;
    if (graph->pattern != NCCL_TOPO_PATTERN_NVLS) {
      netId = graph->inter[channel*2+index];
    } else {
      NCCLCHECK(getNvlsNetDev(comm, graph, channelId, &netId));
    }
    NCCLCHECK(ncclTopoIdToNetDev(comm->topo, netId, &netDev));
    if (dev) *dev = netDev;
    if (id) *id = netId;
    NCCLCHECK(ncclTopoGetIntermediateRank(comm->topo, rank, netId, proxyRank));
  } else if (peerRank == -1) {
    return ncclInternalError;
  } else {
    // Start with our local NIC and local Rank
    NCCLCHECK(ncclTopoGetLocalNet(comm->topo, rank, channelId, &netId, &netDev));
    if (dev) *dev = netDev;
    if (id) *id = netId;
    *proxyRank = rank;

    int pxnLevel = ncclPxnDisable(comm) == 1 ? 0 : ncclParamP2pPxnLevel();
    // See whether we can use the remote rank preferred device.
    if (ncclParamCrossNic() == 0 || (pxnLevel != 0)) {
      // Find local NIC number close to local nvmlDev
      int nvmlDev = comm->peerInfo[peerRank].nvmlDev;
      int localRank;
      if (ncclTopoDevToRank(comm->topo, nvmlDev, &localRank) != ncclSuccess) return ncclSuccess;
      NCCLCHECK(ncclTopoGetLocalNet(comm->topo, localRank, channelId, &netId, &netDev));

      // Check that device exists on our node
      if (ncclParamCrossNic() == 0) {
        if (dev) *dev = netDev;
        if (id) *id = netId;
      }
      if (pxnLevel == 1) {
        int g, n;
        NCCLCHECK(ncclTopoRankToIndex(comm->topo, rank, &g));
        NCCLCHECK(ncclTopoIdToIndex(comm->topo, NET, netId, &n));
        struct ncclTopoNode* gpu = comm->topo->nodes[GPU].nodes+g;
        if (gpu->paths[NET][n].type <= PATH_PXN) {
          if (dev) *dev = netDev;
          if (id) *id = netId;
          NCCLCHECK(ncclTopoGetIntermediateRank(comm->topo, rank, *dev, proxyRank));
        }
      } else if (pxnLevel == 2) {
        // Check which local GPU corresponds to that NIC and see if we can use PXN.
        int n, g1, g2;
        NCCLCHECK(ncclTopoIdToIndex(comm->topo, NET, netId, &n));
        NCCLCHECK(ncclTopoRankToIndex(comm->topo, rank, &g1));
        NCCLCHECK(ncclTopoGetLocalGpu(comm->topo, netId, &g2));
        if (g2 != -1) {
          struct ncclTopoNode* peerGpu = comm->topo->nodes[GPU].nodes+g2;
          if (peerGpu->paths[GPU][g1].type <= PATH_NVL && peerGpu->paths[NET][n].type <= PATH_PXB) {
            *proxyRank = peerGpu->gpu.rank;
            if (dev) *dev = netDev;
            if (id) *id = netId;
            return ncclSuccess;
          }
        }
      }
    }
  }
  return ncclSuccess;
}
