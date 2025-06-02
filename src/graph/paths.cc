/*************************************************************************
 * Copyright (c) 2018-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "graph.h"
#include "topo.h"
#include "comm.h"
#include "net.h"
#include "channel.h"
#include "transport.h"
#include "device.h"

// Pre-compute GPU->NIC, GPU->GPU and NIC->GPU paths

struct ncclTopoNodeList {
  struct ncclTopoNode* list[NCCL_TOPO_MAX_NODES];
  int count;
};
//根据节点类型 t 和节点 id id ，在 topoSystem 中找到目标节点在 node->paths 里的路径指针，并返回给调用者。
//其实主要就是找到指针，因为我们要知道这个path的指针指向的是谁，而系统中其实已经创建了相关的
// 这里就是从当前node出发，到id这个节点的路径。
//这样设计的好处是可以用统一的方式，快速定位和访问任意节点到任意类型、任意编号节点的路径信息，方便路径的查找、更新和管理。
static ncclResult_t getPath(struct ncclTopoSystem* system, struct ncclTopoNode* node, int t, int64_t id, struct ncclTopoLinkList** path) {
  for (int i=0; i<system->nodes[t].count; i++) {
    // node->paths[t] 就是一个指向“所有 t 类型节点路径数组”的指针。
    //i是在系统中类型为 t 的节点的编号（即 system->nodes[t].nodes[i] 就是第 i 个 t 类型节点）。
    //node->paths[t]+i 就是“从 node 出发，到系统中第 i 个 t 类型节点的路径信息结构体（ncclTopoLinkList）”的指针。
    //假设 node 是 GPU0，t=NET，i=1，那么 node->paths[NET]+1 就是“从 GPU0 到系统中第 1 个 NET 节点的最优路径信息”的指针。

    if (system->nodes[t].nodes[i].id == id) {
      *path = node->paths[t]+i;
      return ncclSuccess;
    }
  }
  WARN("Could not find node of type %d id %lx", t, id);
  return ncclInternalError;
}

NCCL_PARAM(NvbDisable, "NVB_DISABLE", 0);
//该函数以 baseNode 为起点，使用 BFS 算法，为系统中所有其他节点建立到baseNode的最优路径（带宽最大、跳数最少），并记录路径类型。
/*
- 最短跳数优先 ：优先选择跳数更少的路径（即 BFS 的天然特性）。
- 带宽最大优先 ：在跳数相同的情况下，选择带宽更大的路径。
- 避免环路和重复 ：每个节点只在更优条件下才会被加入下一轮 BFS 队列，避免重复遍历。
spfa算法
*/
static ncclResult_t ncclTopoSetPaths(struct ncclTopoNode* baseNode, struct ncclTopoSystem* system) {
    // 如果还没有为 baseNode 分配 paths 数组，则分配并初始化为 PATH_DIS（不可达）
  if (baseNode->paths[baseNode->type] == NULL) {
    // 注意这里是同类型节点。
    NCCLCHECK(ncclCalloc(baseNode->paths+baseNode->type, system->nodes[baseNode->type].count));
    for (int i=0; i<system->nodes[baseNode->type].count; i++) baseNode->paths[baseNode->type][i].type = PATH_DIS;
  }
  // 使用 BFS 进行路径搜索
  // breadth-first search to set all paths to that node in the system
  struct ncclTopoNodeList nodeList;
  struct ncclTopoNodeList nextNodeList = { { 0 }, 0 };
  nodeList.count = 1; nodeList.list[0] = baseNode;
  struct ncclTopoLinkList* basePath;// 在这里代表的是“baseNode 到自身的路径信息”，即从 baseNode 出发，回到 baseNode 本身的路径（也就是自环/self-loop）。
  //NCCL 的路径信息是以二维数组（paths）形式存储的，每个节点到每种类型、每个编号节点的路径都单独有一份结构体
  // 可以精准地拿到 baseNode 到自身的路径结构体指针，然后对其进行初始化
  NCCLCHECK(getPath(system, baseNode, baseNode->type, baseNode->id, &basePath));
  basePath->count = 0;// 到自身的路径长度为0
  basePath->bw = LOC_BW;// 到自身的带宽为本地带宽
  basePath->type = PATH_LOC;// 路径类型为本地
  // BFS 主循环
  while (nodeList.count) {
    nextNodeList.count = 0;
    for (int n=0; n<nodeList.count; n++) {
      struct ncclTopoNode* node = nodeList.list[n];
      struct ncclTopoLinkList* path;//获取当前节点到原点的路径。
      
      NCCLCHECK(getPath(system, node, baseNode->type, baseNode->id, &path));
      //遍历所有的边
      for (int l=0; l<node->nlinks; l++) {
        struct ncclTopoLink* link = node->links+l;//当前连接
        struct ncclTopoNode* remNode = link->remNode;// 当前连接到的点。
        // 如果 remNode 还没有分配 paths，则分配并初始化为不可达
        if (remNode->paths[baseNode->type] == NULL) {
          NCCLCHECK(ncclCalloc(remNode->paths+baseNode->type, system->nodes[baseNode->type].count));
          for (int i=0; i<system->nodes[baseNode->type].count; i++) remNode->paths[baseNode->type][i].type = PATH_DIS;
        }
        struct ncclTopoLinkList* remPath;
        //获取 remNode 到原点的路径。
        NCCLCHECK(getPath(system, remNode, baseNode->type, baseNode->id, &remPath));
        float bw = std::min(path->bw, link->bw);// 路径带宽取当前路径和新链路的最小值

        // allow routing through a GPU only as 1 hop
        // 只允许通过 GPU 作为 1 跳 NVLink 路由（即只允许 GPU 作为中转节点时，必须是直接通过 NVLink 连接的 GPU，并且不能是多跳路径）。
        // 简单来说就是只允许直连的gpu路径（前提是这些条件成立）
        // 从其他节点到某个gpuB中间最多经过一个和gpuB直连的gpuA
        if (node != baseNode && node->type == GPU &&
            (ncclParamNvbDisable() || link->type != LINK_NVL || remNode->type != GPU || path->count > 1)) continue;
        // 如果 remPath 尚未设置，或当前路径更短且带宽更大，则更新 remPath
        //虽然 BFS 保证了第一次访问到的路径是最短的，但由于 NCCL 拓扑的特殊性（多链路、多类型），同一节点可能被多次访问。
        /*
        例如：
          - GPU0 通过 NVLink 访问 GPU1（跳数1，带宽高），这是第一次访问，记录下来。
          - 后续 BFS 过程中，GPU0 还可能通过 PCIe 间接访问 GPU1（跳数2，带宽低），这时虽然节点还是 GPU1，但路径不同。
          没有全局 visited 标记，是为了允许同一节点在不同条件下多次入队和路径比较，最终只保留最优路径。
          代码通过遍历 nextNodeList 实现了“本轮不重复入队”。
        */
        if ((remPath->bw == 0 || remPath->count > path->count) && remPath->bw < bw) {
          // Find reverse link
          for (int l=0; l<remNode->nlinks; l++) {
            //在 remNode（当前遍历到的节点）所有的出边（links）中，找到一条“指向 node 且类型与 link 相同”的边。
            //如果找到这样一条边，就把它的指针赋值给 remPath->list[0] ，表示 remPath 路径的第一跳就是这条反向边。
            // 如果直接用 node->links[l]，那这个指针是“从 node 出发到 remNode”的， 但我们现在要构建的是“从 remNode 出发到 baseNode”的路径， 
            // 所以 第一跳必须是 remNode 出发的链路指针 ，即 remNode->links[x]。因为虽然本质上是一样的边，但是在内存中是不同的两个指针。
            if (remNode->links[l].remNode == node && remNode->links[l].type == link->type) {
              remPath->list[0] = remNode->links+l;
              break;
            }
          }
          if (remPath->list[0] == NULL) {
            WARN("Failed to find reverse path from remNode %d/%lx nlinks %d to node %d/%lx",
                 remNode->type, remNode->id, remNode->nlinks, node->type, node->id);
            return ncclInternalError;
          }
          // Copy the rest of the path
          // 拷贝之前路径上的 link，形成完整路径
          for (int i=0; i<path->count; i++) remPath->list[i+1] = path->list[i];
          remPath->count = path->count + 1;
          remPath->bw = bw;
// 路径类型推断
          // Start with path type = link type. PATH and LINK types are supposed to match.
          // Don't consider LINK_NET as we only care about the NIC->GPU path.
          // NCCL 在路径类型推断时，实际上只关心 GPU/NIC 之间的本地路径（比如 GPU 到 NIC 的直连），而不关心网络链路本身的类型。
          // 
          int type = link->type == LINK_NET ? LINK_LOC : link->type;
          // Differentiate between one and multiple PCI switches
          if (node->type == PCI && remNode->type == PCI) type = PATH_PXB; // 多级 PCIe Switch
          // Consider a path going through the CPU as PATH_PHB
          // NCCL 主要关注 GPU/NIC 之间的通信优化，CPU <-> CPU 的路径在实际通信中很少直接用到，所以这里做了简化处理
          // 即使 CPU <-> CPU 被归为 PATH_PHB，实际带宽/延迟建模和调度时影响不大，因为 NCCL 主要不会让数据在 CPU 之间直接流动。
          if (link->type == LINK_PCI && (node->type == CPU || link->remNode->type == CPU)) type = PATH_PHB; // 经过 CPU Host Bridge
          // Set 1 hop NVLink as NVB
          if (node->type == GPU && path->type == PATH_NVL && type == PATH_NVL && remPath->count > 1) type = PATH_NVB; // 多跳 NVLink

          remPath->type = std::max(path->type, type);

          // Add to the list for the next iteration if not already in the list
          // 将 remNode 加入下一轮 BFS 队列（避免重复）
          int i;
          for (i=0; i<nextNodeList.count; i++) if (nextNodeList.list[i] == remNode) break;
          if (i == nextNodeList.count) nextNodeList.list[nextNodeList.count++] = remNode;
        }
      }
    }
    memcpy(&nodeList, &nextNodeList, sizeof(nodeList));
  }
  return ncclSuccess;
}

static void printNodePaths(struct ncclTopoSystem* system, struct ncclTopoNode* node) {
  const int linesize = 1024;
  char line[linesize];
#ifdef ENABLE_TRACE
  INFO(NCCL_GRAPH, "Paths from %s/%lx-%lx :", topoNodeTypeStr[node->type], NCCL_TOPO_ID_SYSTEM_ID(node->id), NCCL_TOPO_ID_LOCAL_ID(node->id));
#else
  snprintf(line, linesize, "%s/%lx-%lx :", topoNodeTypeStr[node->type], NCCL_TOPO_ID_SYSTEM_ID(node->id), NCCL_TOPO_ID_LOCAL_ID(node->id));
  int offset = strlen(line);
#endif
  for (int t=0; t<NCCL_TOPO_NODE_TYPES; t++) {
    if (node->paths[t] == NULL) continue;
    for (int n = 0; n<system->nodes[t].count; n++) {
#ifdef ENABLE_TRACE
      line[0] = 0;
      int offset = 0;
      for (int i=0; i<node->paths[t][n].count; i++) {
        struct ncclTopoLink* link = node->paths[t][n].list[i];
        struct ncclTopoNode* remNode = link->remNode;
        snprintf(line+offset, linesize-offset, "--%s(%g)->%s/%lx-%lx", topoLinkTypeStr[link->type], link->bw, topoNodeTypeStr[remNode->type], NCCL_TOPO_ID_SYSTEM_ID(remNode->id), NCCL_TOPO_ID_LOCAL_ID(remNode->id));
        offset = strlen(line);
      }
      INFO(NCCL_GRAPH, "%s (%f)", line, node->paths[t][n].bw);
#else
      snprintf(line+offset, linesize-offset, "%s/%lx-%lx (%d/%.1f/%s) ", topoNodeTypeStr[t], NCCL_TOPO_ID_SYSTEM_ID(system->nodes[t].nodes[n].id), NCCL_TOPO_ID_LOCAL_ID(system->nodes[t].nodes[n].id), node->paths[t][n].count, node->paths[t][n].bw, topoPathTypeStr[node->paths[t][n].type]);
      offset = strlen(line);
#endif
    }
  }
#ifndef ENABLE_TRACE
  INFO(NCCL_GRAPH, "%s", line);
#endif
}

ncclResult_t ncclTopoPrintPaths(struct ncclTopoSystem* system) {
  for (int i=0; i<system->nodes[GPU].count; i++) {
    printNodePaths(system, system->nodes[GPU].nodes+i);
  }
  for (int i=0; i<system->nodes[NET].count; i++) {
    printNodePaths(system, system->nodes[NET].nodes+i);
  }
  return ncclSuccess;
}
//在gpu到达所有cpu的路径中，找到路径最短的cpu，作为localCpu，也就是找到最近的cpu
ncclResult_t ncclGetLocalCpu(struct ncclTopoSystem* system, int gpu, int* retCpu) {
  // Find the closest CPU to a GPU
  int minHops = 0;
  int localCpu = -1;

  struct ncclTopoLinkList* paths = system->nodes[GPU].nodes[gpu].paths[CPU];
  for (int c=0; c<system->nodes[CPU].count; c++) {
    int hops = paths[c].count;
    if (hops > 0 && (minHops == 0 || hops < minHops)) {
      localCpu = c;
      minHops = hops;
    }
  }
  if (localCpu == -1) {
    WARN("Error : could not find CPU close to GPU %d", gpu);
    return ncclInternalError;
  }
  *retCpu = localCpu;
  return ncclSuccess;
}
//，构建一条“经过中间节点（通常是CPU或GPU）”的复合路径。例如：从节点1经过CPU再到节点2，把这两段路径拼接起来，形成一条完整的路径，并更新路径的类型、带宽等属性。
// 不过这里有点疑问，为什么没有更新节点2到节点1的路径呢？因为最优路径可能不同？
static ncclResult_t addInterStep(struct ncclTopoSystem* system, int tx, int ix, int t1, int i1, int t2, int i2) {
  struct ncclTopoNode* cpuNode = system->nodes[tx].nodes+ix;
  struct ncclTopoNode* srcNode = system->nodes[t1].nodes+i1;

  int l=0;
  // Node 1 -> CPU
  for (int i=0; i<srcNode->paths[tx][ix].count; i++) srcNode->paths[t2][i2].list[l++] = srcNode->paths[tx][ix].list[i];
  // CPU -> Node 2
  for (int i=0; i<cpuNode->paths[t2][i2].count; i++) srcNode->paths[t2][i2].list[l++] = cpuNode->paths[t2][i2].list[i];

  // Update path characteristics
  srcNode->paths[t2][i2].count = l;
  srcNode->paths[t2][i2].type = std::max(srcNode->paths[tx][ix].type, cpuNode->paths[t2][i2].type);
  //如果中间节点是GPU，则特殊标记为 PATH_PXN 。
  if (tx == GPU) srcNode->paths[t2][i2].type = PATH_PXN;
  srcNode->paths[t2][i2].bw = std::min(srcNode->paths[tx][ix].bw, cpuNode->paths[t2][i2].bw);
  return ncclSuccess;
}

// Remove/free all paths
static void ncclTopoRemovePaths(struct ncclTopoSystem* system) {
  for (int t1=0; t1<NCCL_TOPO_NODE_TYPES; t1++) {
    for (int n=0; n<system->nodes[t1].count; n++) {
      struct ncclTopoNode* node = system->nodes[t1].nodes+n;
      for (int t2=0; t2<NCCL_TOPO_NODE_TYPES; t2++) {
        if (node->paths[t2]) free(node->paths[t2]);
        node->paths[t2] = NULL;
      }
    }
  }
}

static const int levelsOldToNew[] = { PATH_LOC, PATH_PIX, PATH_PXB, PATH_PHB, PATH_SYS, PATH_SYS };
ncclResult_t ncclGetLevel(int* level, const char* disableEnv, const char* levelEnv) {
  if (*level == -1) {
    int l = -1;
    if (disableEnv) {
      const char* str = ncclGetEnv(disableEnv);
      if (str) {
        int disable = strtol(str, NULL, 0);
        if (disable == 1) l = 0;
        if (l >= 0) INFO(NCCL_ALL, "%s set by environment to %d", disableEnv, disable);
      }
    }
    if (l == -1) {
      const char* str = ncclGetEnv(levelEnv);
      if (str) {
        for (int i=0; i<=PATH_SYS; i++) {
          if (strcmp(str, topoPathTypeStr[i]) == 0) {
            l = i;
            break;
          }
        }
        // Old style numbering
        // levelsOldToNew to is an array with each index corresponding to the
        // "old level" int, and each value mapping to the correct value defined in topo.h
        // maxOldLevel is a quick check to handle out of bounds (based on the length of levelsOldToNew)
        if (l == -1 && str[0] >= '0' && str[0] <= '9') {
          int oldLevel = strtol(str, NULL, 0);
          const int maxOldLevel = sizeof(levelsOldToNew)/sizeof(int) - 1;
          if (oldLevel > maxOldLevel) oldLevel = maxOldLevel;
          l = levelsOldToNew[oldLevel];
        }
        if (l >= 0) INFO(NCCL_ALL, "%s set by environment to %s", levelEnv, topoPathTypeStr[l]);
      }
    }
    *level = l >= 0 ? l : -2;
  }
  return ncclSuccess;
}

NCCL_PARAM(IgnoreDisabledP2p, "IGNORE_DISABLED_P2P", 0);

// 用于判断两个GPU之间是否可以直接P2P通信（包括NVLink/PCIe等），并根据拓扑、环境变量、硬件状态等多重条件做出决策。
int ncclTopoUserP2pLevel = -1;
ncclResult_t ncclTopoCheckP2p(struct ncclComm* comm, struct ncclTopoSystem* system, int rank1, int rank2,
                              int* p2p, int *read, int* intermediateRank) {
  int mnnvl = 0;
  struct ncclPeerInfo* info1 = NULL;
  struct ncclPeerInfo* info2 = NULL;
  *p2p = 0;// 默认不支持P2P
  if (read) *read = 0;// 默认不支持P2P Read
  if (intermediateRank) *intermediateRank = -1;

  // Rule out different nodes / isolated containers
  // 排除不同物理节点/容器的情况
  if (comm) {
    info1 = comm->peerInfo+rank1;
    info2 = comm->peerInfo+rank2;
    if (info1->hostHash != info2->hostHash) {// 不同主机
      if (comm->MNNVL) {
        NCCLCHECK(ncclTopoCheckMNNVL(comm->topo, info1, info2, &mnnvl)); // 检查是否在同一NVLink Fabric
        if (!mnnvl) return ncclSuccess;// 不在同一Fabric，直接返回, 不支持P2P。
      } else {
        return ncclSuccess;
      }
    } else if (info1->shmDev != info2->shmDev) {// 共享内存设备不同，也不支持P2P
      /*
      shmDev 通常代表 GPU 所在的“共享内存设备”或“NUMA 节点”或“同一 CPU socket 下的设备”。如果两个 GPU 不属于同一个 shmDev，
      说明它们之间即使有物理互联，操作系统层面也可能不允许直接的共享内存访问，或者性能会很差，甚至根本无法建立 P2P 通道。
      */
      return ncclSuccess;
    }
  }
// 从拓扑结构中获取两个GPU的索引
  // Get GPUs from topology
  int g1, g2;
  NCCLCHECK(ncclTopoRankToIndex(system, rank1, &g1));
  struct ncclTopoNode* gpu1 = system->nodes[GPU].nodes+g1;
  if (ncclTopoRankToIndex(system, rank2, &g2) == ncclInternalError) {
    // GPU not found, we can't use p2p. // 找不到GPU，不能用P2P
    return ncclSuccess;
  }
// 如果路径是两跳，且中间节点是GPU，则记录中转GPU的索引
  int intermediateIndex = -1;
  // Set intermediate GPU rank, if routing through an intermediate GPU.
  struct ncclTopoLinkList* path = gpu1->paths[GPU]+g2;
  if (path->count == 2) {
    struct ncclTopoNode* intermediateNode = path->list[0]->remNode;
    if (intermediateNode->type == GPU) {// 中间节点是GPU
      intermediateIndex = intermediateNode - system->nodes[GPU].nodes;
      if (intermediateRank) *intermediateRank = intermediateNode->gpu.rank;
    }
  }
 // 默认P2P只允许跨PCIe桥（不允许跨Host Bridge/CPU）
  // By default don't use P2P across CPU Host Bridges and further apart
  int p2pLevel = PATH_PXB;

  int arch, vendor, model;
  NCCLCHECK(ncclTopoCpuType(system, &arch, &vendor, &model));
  // Allow P2P between pairs of GPUs on AMD systems
  // AMD平台且只有2块GPU时，放宽P2P限制
  // 可能是因为AMD 的部分服务器主板设计，两个 GPU 可能分别挂在不同的 CPU（NUMA 节点）下，彼此之间的直连路径必然要经过 CPU Host Bridge。
  //在实际部署中，2块GPU很可能分别插在不同的CPU（NUMA节点）下的PCIe插槽上，这样GPU之间的P2P通信路径就必然要经过CPU Host Bridge（即跨NUMA节点）。
  if ((arch == NCCL_TOPO_CPU_ARCH_X86 && vendor == NCCL_TOPO_CPU_VENDOR_AMD) && system->nodes[GPU].count <= 2) p2pLevel = PATH_SYS;

  // User override // 用户环境变量覆盖 初始值为 -1，表示尚未设置
  if (ncclTopoUserP2pLevel == -1)
  //尝试从环境变量 NCCL_P2P_DISABLE 或 NCCL_P2P_LEVEL 读取用户设置的P2P等级。
//如果环境变量没设置，或者内容非法， ncclTopoUserP2pLevel 会被设置为 -2。
    NCCLCHECK(ncclGetLevel(&ncclTopoUserP2pLevel, "NCCL_P2P_DISABLE", "NCCL_P2P_LEVEL"));
  if (ncclTopoUserP2pLevel != -2) {//说明用户确实通过环境变量指定了P2P等级，这时就用用户指定的等级覆盖默认的 p2pLevel ，并跳到 compare 标签，直接用这个等级进行后续判断。
    p2pLevel = ncclTopoUserP2pLevel;
    goto compare;// 跳到比较阶段
  }


compare:
  // Compute the PCI distance and compare with the p2pLevel.
  // 比较路径类型与p2pLevel，决定是否支持P2P
  if (path->type <= p2pLevel) *p2p = 1;
//这段代码是NCCL多重保障机制中的最后一道防线，确保P2P通信既符合拓扑和策略，也符合底层硬件/驱动的实际能力，最大程度保证通信的安全和稳定。
  if (*p2p == 1) {
    
    // NCCL_IGNORE_DISABLED_P2P=2 is used by unit tests that don't want to
    // validate against NVML at all since they are pretending to be on other hw.
    // NCCL_IGNORE_DISABLED_P2P=2 用于单元测试，跳过NVML校验
    //不是同一块GPU,要么没有通信上下文（comm），要么这两个GPU都在同一主机上。环境变量 NCCL_IGNORE_DISABLED_P2P 没有被设置为2（2通常用于单元测试，跳过NVML校验）。
    //这段代码是NCCL在决定是否允许两个GPU之间P2P通信的最后一步安全检查，确保硬件和驱动层面（NVML）也支持P2P
    //comm == NULL 的判断是为了兼容 NCCL 在没有通信上下文时的调用场景，保证代码的健壮性和通用性。此时直接允许后续的 P2P 能力检查，不做主机一致性校验。
    if (g1 != g2 && (comm == NULL || (info1->hostHash == comm->peerInfo[comm->rank].hostHash &&
                                      info1->hostHash == info2->hostHash)) && ncclParamIgnoreDisabledP2p() != 2) {
      int indexes[3] = {-1,-1,-1};//依次存放起点GPU、中间节点（如果有）、终点GPU的设备号。 verticeN 表示实际用到的节点数。
      int verticeN = 0;
      NCCLCHECK(ncclNvmlEnsureInitialized());

      indexes[verticeN++] = system->nodes[GPU].nodes[g1].gpu.dev;
      if (intermediateIndex != -1) indexes[verticeN++] = system->nodes[GPU].nodes[intermediateIndex].gpu.dev;
      indexes[verticeN++] = system->nodes[GPU].nodes[g2].gpu.dev;
// 检查NVML层面P2P状态 。遍历路径上的每一段（比如A->B->C会检查A-B和B-C）
      for (int i=1; i < verticeN; i++) {
        nvmlGpuP2PStatus_t status;
        //通过 ncclNvmlDevicePairs 查询NVML层面这两个GPU之间的P2P读写状态（ p2pStatusRead 和 p2pStatusWrite ）。
        status = ncclNvmlDevicePairs[indexes[i-1]][indexes[i-0]].p2pStatusRead;
        bool good = status == NVML_P2P_STATUS_OK;
        status = ncclNvmlDevicePairs[indexes[i-1]][indexes[i-0]].p2pStatusWrite;
        good &= status == NVML_P2P_STATUS_OK;//只有读写都为 NVML_P2P_STATUS_OK ，才认为这段P2P是可用的。
        if (!good) {//如果NVML报告P2P不可用（ good == false ）
          if (!ncclParamIgnoreDisabledP2p()) {//如果不强制忽略NVML的P2P禁用
            if (path->type <= PATH_NVB) {
              //说明本应支持P2P但被禁用
              WARN("P2P is disabled between NVLINK connected GPUs %d and %d. This should not be the case given their connectivity, and is probably due to a hardware issue. If you still want to proceed, you can set NCCL_IGNORE_DISABLED_P2P=1.", indexes[i-1], indexes[i-0]);
              return ncclUnhandledCudaError;
            } else if (path->type < PATH_SYS) {
              INFO(NCCL_INIT, "P2P is disabled between connected GPUs %d and %d. You can repress this message with NCCL_IGNORE_DISABLED_P2P=1.", indexes[i-1], indexes[i-0]);
            }
          }
          *p2p = 0;// NVML不支持P2P，最终不允许
        }
      }
    }
  }
  // 如果是NVLink直连，且两块GPU都是Ampere架构，允许P2P Read
  /*
  - 因为P2P Write的兼容性和性能在绝大多数平台上都更好、更稳定，是NCCL默认优先支持的方式。
- P2P Read只有在特定平台（如Ampere+NVLink）才被认为是安全且高效的，NCCL会做更严格的判断，只有满足条件才允许开启。
在硬件实现和驱动支持上，P2P Write 通常比 P2P Read 更容易实现和优化。很多早期的NVIDIA GPU和主板/驱动只支持P2P Write，不支持P2P Read，或者P2P Read性能很差。

  */
  if (path->type == PATH_NVL) {
    struct ncclTopoNode* gpu2 = system->nodes[GPU].nodes+g2;
    // Enable P2P Read for Ampere/NVLink only
    if (read && (gpu1->gpu.cudaCompCap == gpu2->gpu.cudaCompCap) && (gpu1->gpu.cudaCompCap == 80)) *read = 1;
  }

  return ncclSuccess;
}

// MNNVL: Check whether peers are in the same fabric cluster and clique
ncclResult_t ncclTopoCheckMNNVL(struct ncclTopoSystem* system, struct ncclPeerInfo* info1, struct ncclPeerInfo* info2, int* ret) {
  *ret = 0;

  nvmlGpuFabricInfoV_t *fabricInfo1 = &info1->fabricInfo;
  nvmlGpuFabricInfoV_t *fabricInfo2 = &info2->fabricInfo;
  // A zero UUID means we don't have MNNVL fabric info
  if ((((long *)&fabricInfo2->clusterUuid)[0]|((long *)fabricInfo2->clusterUuid)[1]) == 0) return ncclSuccess;
  if ((memcmp(fabricInfo1->clusterUuid, fabricInfo2->clusterUuid, NVML_GPU_FABRIC_UUID_LEN) == 0) &&
      (fabricInfo1->cliqueId == fabricInfo2->cliqueId)) {
    INFO(NCCL_NET, "MNNVL matching peer 0x%lx UUID %lx.%lx cliqueId 0x%x",
         info2->busId, ((long *)fabricInfo2->clusterUuid)[0], ((long *)fabricInfo2->clusterUuid)[1], fabricInfo2->cliqueId);
    *ret = 1;
  }
  return ncclSuccess;
}

NCCL_PARAM(NetGdrRead, "NET_GDR_READ", -2);
int ncclTopoUserGdrLevel = -1;
//用于判断某个 GPU 和某个网卡（NET）之间是否可以启用 GDR（GPU Direct RDMA），即网卡能否直接访问 GPU 显存进行高效数据传输。
ncclResult_t ncclTopoCheckGdr(struct ncclTopoSystem* system, int rank, int64_t netId, int read, int* useGdr) {
  *useGdr = 0;

  // Get GPU and NET
  int n, g;
  NCCLCHECK(ncclTopoIdToIndex(system, NET, netId, &n));
  struct ncclTopoNode* net = system->nodes[NET].nodes+n;
  NCCLCHECK(ncclTopoRankToIndex(system, rank, &g));
  struct ncclTopoNode* gpu = system->nodes[GPU].nodes+g;

  // Check that both the NIC and GPUs support it 如果网卡或 GPU 本身不支持 GDR，直接返回
  if (net->net.gdrSupport == 0) return ncclSuccess;
  if (gpu->gpu.gdrSupport == 0) return ncclSuccess;
  //如果是 GDR Read（即网卡向 GPU 读数据）
  if (read) { // For reads (sends) only enable under certain conditions
    int gdrReadParam = ncclParamNetGdrRead();
    if (gdrReadParam == 0) return ncclSuccess;
    // Disable GDR Reads pre-Ampere when we have other PCI flows
    // 对于 Ampere 之前的 GPU（算力 < 80），如果没有 NVLink 直连，禁用 GDR Read（因为性能不佳或不稳定）。
    if (gdrReadParam < 0 && gpu->gpu.cudaCompCap < 80) {
      int nvlink = 0;
      // Since we don't know whether there are other communicators,
      // it's better to keep things local if we have a single GPU.
      // 由于我们无法确定系统中是否还存在其他通信器（communicator，NCCL 的通信上下文），
      // 如果当前只有一块 GPU，最好让所有操作都局限在本地（local），即只在这块 GPU 上进行，不要尝试跨设备或跨节点通信。
      if (system->nodes[GPU].count == 1) nvlink = 1; //如果系统中只有一块 GPU，则直接认为“有 NVLink 直连”，即允许本地 GDR Read。
      for (int i=0; i<system->nodes[GPU].count; i++) {
        if (i == g) continue;
        //这里的 NVLink 检查其实是 NCCL 用来“推测”平台整体互联和 GDR 能力的一个“信号”。有 NVLink 说明平台较新、互联好，GDR Read 风险小，所以允许；否则就禁用。
        if (gpu->paths[GPU][i].type == PATH_NVL) { //如果有多块 GPU，遍历所有其他 GPU，检查当前 GPU 是否与其他 GPU 之间存在 NVLink 直连
          nvlink = 1;
          break;
        }
      }
      if (!nvlink) return ncclSuccess;
    }
  }
  //检查距离限制
  // Check if we are close enough that it makes sense to enable GDR 
  int netGdrLevel = PATH_PXB;//默认只允许距离不超过 PATH_PXB（PCIe Switch 直连）的路径使用 GDR。
  //用户可以通过环境变量 NCCL_NET_GDR_LEVEL 覆盖默认距离限制。
  NCCLCHECK(ncclGetLevel(&ncclTopoUserGdrLevel, NULL, "NCCL_NET_GDR_LEVEL"));
  if (ncclTopoUserGdrLevel != -2) netGdrLevel = ncclTopoUserGdrLevel;
  int distance = gpu->paths[NET][n].type;
  if (distance == PATH_PXN) {//如果路径类型为 PXN（即通过 peer GPU 中转），则用中转 GPU 到网卡的距离重新判断。
    // In case of PXN, use the intermediate GPU distance instead
    int proxyRank, g;
    NCCLCHECK(ncclTopoGetIntermediateRank(system, gpu->gpu.rank, netId, &proxyRank));
    NCCLCHECK(ncclTopoRankToIndex(system, proxyRank, &g));
    struct ncclTopoNode* proxyGpu = system->nodes[GPU].nodes+g;
    distance = proxyGpu->paths[NET][n].type;
  }
  if (distance > netGdrLevel) {
    INFO(NCCL_NET,"GPU Direct RDMA Disabled for GPU %d / HCA %lx (distance %d > %d)", rank, netId, distance, netGdrLevel);
    return ncclSuccess;
  }

  *useGdr = 1;
  INFO(NCCL_NET,"GPU Direct RDMA Enabled for GPU %d / HCA %lx (distance %d <= %d), read %d", rank, netId, distance, netGdrLevel, read);
  return ncclSuccess;
}

ncclResult_t ncclTopoIsGdrAvail(struct ncclTopoSystem* system, int rank, bool *avail) {
  int netNum = system->nodes[NET].count;
  int useGdr = 0;
  *avail = false;
  for (int n = 0; n < netNum; n++) {
    int64_t netId = system->nodes[NET].nodes[n].id;
    NCCLCHECK(ncclTopoCheckGdr(system, rank, netId, 1, &useGdr));
    if (useGdr) {
      *avail = true;
      break;
    }
    NCCLCHECK(ncclTopoCheckGdr(system, rank, netId, 0, &useGdr));
    if (useGdr) {
      *avail = true;
      break;
    }
  }
  return ncclSuccess;
}

// Set to 0 to disable the flush on Hopper when using GDR
NCCL_PARAM(NetForceFlush, "NET_FORCE_FLUSH", 0);

// Determine whether we need to flush the GDR recv buffers
ncclResult_t ncclTopoNeedFlush(struct ncclComm* comm, int netDev, int rank, int* flush) {
  *flush = 1;
  ncclNetProperties_t props;
  NCCLCHECK(comm->ncclNet->getProperties(netDev, &props));
  if (props.forceFlush == 1 || ncclParamNetForceFlush()) return ncclSuccess;
  int g;
  struct ncclTopoSystem* system = comm->topo;
  NCCLCHECK(ncclTopoRankToIndex(system, rank, &g));
  struct ncclTopoNode* gpu = system->nodes[GPU].nodes+g;
  // Flush is required on Ampere and earlier
  if (gpu->gpu.cudaCompCap >= 90) *flush = 0;
  return ncclSuccess;
}

NCCL_PARAM(NetDisableIntra, "NET_DISABLE_INTRA", 0);

// Check whether going through the network would be faster than going through P2P/SHM.
ncclResult_t ncclTopoCheckNet(struct ncclTopoSystem* system, int rank1, int rank2, int* net) {
  if (ncclParamNetDisableIntra() == 1) {
    *net = 0;
    return ncclSuccess;
  }
  *net = 1;
  // First check the current GPU-to-GPU speed.
  int g1, g2;
  if (ncclTopoRankToIndex(system, rank1, &g1) != ncclSuccess ||
      ncclTopoRankToIndex(system, rank2, &g2) != ncclSuccess) {
    return ncclSuccess;
  }

  struct ncclTopoNode* gpu1 = system->nodes[GPU].nodes+g1;
  struct ncclTopoNode* gpu2 = system->nodes[GPU].nodes+g2;
  float speed = gpu1->paths[GPU][g2].bw;

  // Now check the speed each GPU can access the network through PXB or better
  float netSpeed1 = 0, netSpeed2 = 0;
  for (int n=0; n<system->nodes[NET].count; n++) {
    struct ncclTopoLinkList* path = gpu1->paths[NET]+n;
    if (path->type <= PATH_PXB && path->bw > netSpeed1) netSpeed1 = path->bw;
    path = gpu2->paths[NET]+n;
    if (path->type <= PATH_PXB && path->bw > netSpeed2) netSpeed2 = path->bw;
  }

  if (netSpeed1 > speed && netSpeed2 > speed) return ncclSuccess;
  *net = 0;
  return ncclSuccess;
}
//获取中转gpu
ncclResult_t ncclTopoGetIntermediateRank(struct ncclTopoSystem* system, int rank, int64_t netId, int* intermediateRank) {
  // Get GPU and NET
  int n, g;
  NCCLCHECK(ncclTopoIdToIndex(system, NET, netId, &n));
  NCCLCHECK(ncclTopoRankToIndex(system, rank, &g));
  struct ncclTopoNode* gpu = system->nodes[GPU].nodes+g;
  struct ncclTopoLinkList* path = gpu->paths[NET]+n;
  if (path->type == PATH_PXN) {
    struct ncclTopoNode* node;
    int type = NVS;
    for (int i=0; i<path->count && type == NVS; i++) {
      node = path->list[i]->remNode;
      type = node->type;
    }
    //遍历路径的每一跳，找到第一个类型为 GPU 的节点（即中转 GPU）。
    if (type != GPU) {
      WARN("Could not find intermediate GPU between GPU rank %d and NIC %lx", rank, netId);
      return ncclInternalError;
    }
    *intermediateRank = node->gpu.rank;
  } else {
    *intermediateRank = rank;
  }
  return ncclSuccess;
}

NCCL_PARAM(PxnDisable, "PXN_DISABLE", 0);

// Net v4 plugins don't have non-blocking connect/accept. We can't therefore use
// remote proxies without risking deadlocks
// 该函数用于判断 NCCL 是否需要禁用 PXN（Proxy NIC，中转网卡）功能，并返回禁用标志。
// PXN 是 NCCL 在多机多卡通信中，为了优化 GPU 到 NIC 的路径而引入的一种“通过中转 GPU 访问 NIC”的机制
int ncclPxnDisable(struct ncclComm* comm) {
  static int pxnDisable = -1;
  if (pxnDisable == -1) {
    /*
    如果通信上下文存在，并且网络插件版本为4（v4），则强制禁用 PXN。
     原因是 v4 版本的网络插件 不支持非阻塞的 connect/accept ，如果启用 PXN 可能导致死锁（deadlock），所以直接禁用，并打印日志。
    */
    if (comm && ncclNetVersion(comm) == 4) {
      INFO(NCCL_INIT, "PXN Disabled as plugin is v4");
      pxnDisable = 1;
    } else {
      pxnDisable = ncclParamPxnDisable();
    }
  }
  return pxnDisable;
}

ncclResult_t ncclTopoGetPxnRanks(struct ncclComm* comm, int** intermediateRanks, int* nranks) {
  struct ncclTopoSystem* system = comm->topo;
  *nranks = 0;
  *intermediateRanks = NULL;
  if (system->nodes[NET].count == 0) return ncclSuccess;

  int nr = 0;
  int* ranks = NULL;
  for (int rank=0; rank<comm->nRanks; rank++) {
    int64_t netId;
    int proxyRank;
    NCCLCHECK(ncclTopoGetNetDev(comm, comm->rank, NULL, 0, rank, &netId, NULL, &proxyRank));
    if (proxyRank == comm->rank) continue;
    int useGdr;
    NCCLCHECK(ncclTopoCheckGdr(comm->topo, comm->rank, netId, 1, &useGdr));
    if (useGdr == 0) continue;
    int found = 0;
    for (int r=0; r<nr; r++) {
      if (ranks[r] == proxyRank) found = 1;
    }
    if (!found) {
      NCCLCHECK(ncclRealloc(&ranks, nr, nr+1));
      ranks[nr++] = proxyRank;
    }
  }
  *nranks = nr;
  *intermediateRanks = ranks;
  return ncclSuccess;
}
/*
负责 预计算 NCCL 拓扑结构中所有 GPU、CPU、NIC（网卡）、NVSwitch 之间的最优通信路径 ，为后续通信调度和优化提供基础数据。
它会根据硬件连接、P2P 能力、GDR 能力等，自动调整路径，确保数据传输高效且可达。
*/
ncclResult_t ncclTopoComputePaths(struct ncclTopoSystem* system, struct ncclComm* comm) {
  // Precompute paths between GPUs/NICs.

  // Remove everything in case we're re-computing 先清空所有节点的路径信息，防止重复计算或脏数据。
  ncclTopoRemovePaths(system);

  // Set direct paths to CPUs. We need them in many cases.
  // 2. 为每个 CPU 节点设置其它节点到它的路径
  for (int c=0; c<system->nodes[CPU].count; c++) {
    NCCLCHECK(ncclTopoSetPaths(system->nodes[CPU].nodes+c, system));
  }
 // 3. 为每个 GPU 节点设置到其它 GPU 节点到他路径
  // Set direct paths to GPUs.
  for (int g=0; g<system->nodes[GPU].count; g++) {
    NCCLCHECK(ncclTopoSetPaths(system->nodes[GPU].nodes+g, system));
  }
// 为系统中每一个网络接口（NIC）节点预先计算并存储所有节点到它的最优通信路径
  // Set direct paths to NICs.
  for (int n=0; n<system->nodes[NET].count; n++) {
    NCCLCHECK(ncclTopoSetPaths(system->nodes[NET].nodes+n, system));
  }
// 5. 为每个 NVSwitch 节点设置到其它 节点的最优路径
  // Set direct paths to NVSwitches.
  for (int n=0; n<system->nodes[NVS].count; n++) {
    NCCLCHECK(ncclTopoSetPaths(system->nodes[NVS].nodes+n, system));
  }
// 6. 处理 GPU 间 P2P 不可用的情况，将流量绕行 CPU。如果p2p不可用，则首先增加cpu绕行、
  // Update path for GPUs when we don't want to / can't use GPU Direct P2P
  for (int g=0; g<system->nodes[GPU].count; g++) {
    for (int p=0; p<system->nodes[GPU].count; p++) {
      int p2p;
      NCCLCHECK(ncclTopoCheckP2p(comm, system, system->nodes[GPU].nodes[p].gpu.rank,
                                 system->nodes[GPU].nodes[g].gpu.rank, &p2p, NULL, NULL));
      if (p2p == 0) {
        // Divert all traffic through the CPU
        // 如果 GPU p 到 GPU g 之间 P2P 不可用，则通过本地 CPU 绕行
        int cpu;
        NCCLCHECK(ncclGetLocalCpu(system, g, &cpu));//找到距离g最近的cpu
        NCCLCHECK(addInterStep(system, CPU, cpu, GPU, p, GPU, g));
      }
    }
    //comm为NULL时，无法获取peerInfo等通信相关信息，也就无法判断P2P和SHM的可达性，继续执行会导致空指针或无意义的判断。
    if (comm == NULL) continue;
    // 标记那些既不能通过P2P（Peer-to-Peer）也不能通过SHM（共享内存）通信的GPU对 ，为后续的路径修剪做准备。
    // Remove GPUs we can't (or don't want to) communicate with through P2P or SHM
    struct ncclPeerInfo* dstInfo = comm->peerInfo+system->nodes[GPU].nodes[g].gpu.rank;
    for (int p=0; p<system->nodes[GPU].count; p++) {
      if (p == g) continue;
      struct ncclPeerInfo* srcInfo = comm->peerInfo+system->nodes[GPU].nodes[p].gpu.rank;
      int p2p;
      NCCLCHECK(ncclTransports[TRANSPORT_P2P]->canConnect(&p2p, comm, NULL, srcInfo, dstInfo));
      if (p2p == 0) {
        int shm;
        NCCLCHECK(ncclTransports[TRANSPORT_SHM]->canConnect(&shm, comm, NULL, srcInfo, dstInfo));
        if (shm == 0) {
          // Mark this peer as inaccessible. We'll trim it later.
          // 标记该p到g的路径只能通过网络通信
          system->nodes[GPU].nodes[p].paths[GPU][g].type = PATH_NET;
        }
      }
    }
  }

  // Update paths for NICs (no GPU Direct, PXN, ...)
    // 8. 处理 GPU 到 NIC 的路径优化（PXN、GDR等）
  for (int n=0; n<system->nodes[NET].count; n++) {
    struct ncclTopoNode* netNode = system->nodes[NET].nodes+n;
      // 8.1 检查是否可以通过其他 NVLink 连接的 GPU 中转访问 NIC（PXN）
    for (int g=0; g<system->nodes[GPU].count; g++) {
      // Check whether we can access the NIC through another NVLink-connected GPU (PXN)
      /*
      - 某些情况下，直接从当前 GPU 到目标网卡的路径带宽较低，或者路径类型不理想（比如需要经过 CPU Host Bridge，带宽低、延迟高）。
- 但如果先通过 NVLink 到另一个 GPU（peerNode），再由该 GPU 通过 PCIe 访问网卡，可能能获得更高的带宽或更优的路径类型（比如避免经过 CPU）。
      */
      struct ncclTopoNode* gpu = system->nodes[GPU].nodes+g;
      if (ncclPxnDisable(comm) != 1) {//如果没有禁用PXN功能
        int localGpuIndex;
        NCCLCHECK(ncclTopoGetLocalGpu(system, netNode->id, &localGpuIndex));//查找与当前 net（netNode）相连 GPU（localGpuIndex）
        if (localGpuIndex != g && localGpuIndex != -1) {//如果当前遍历的 GPU（g）不是搜索到的该net连接的GPU，并且确实找到了直连 GPU，才继续。
          // PXN = PCI + NVLink.
          struct ncclTopoNode* peerNode = system->nodes[GPU].nodes+localGpuIndex;//获取连接该 NIC 的 GPU 节点指针
          // Only use PXN for NIC n if remote GPU p ...
          if (peerNode->paths[NET][n].type <= PATH_PXB && // Is connected to the NIC through PCI
              peerNode->paths[GPU][g].type <= PATH_NVL && // Is connected to us through NVLink
              NCCL_TOPO_ID_SYSTEM_ID(peerNode->id) == NCCL_TOPO_ID_SYSTEM_ID(gpu->id) && // Is on the same node as us
              (peerNode->paths[NET][n].bw > gpu->paths[NET][n].bw || // Has either higher BW to that NIC
               gpu->paths[NET][n].type > PATH_PXB))                  // or avoids going through a CPU
          // We can use that GPU as relay to communicate with that NIC.
          // Only enabling it in the GPU->NIC direction for now to favor
          // receiving locally and sending remotely (consistent with net.cc) 明确说明只做单向补全。
          // 通过 peerNode GPU 中转访问 NIC，优化带宽或避免经过 CPU
          /*
          AI解释的这里没有反向的原因是：
          - PXN 的主要目的是 提升 GPU 发送到 NET 的带宽 或优化路径类型（比如避免经过 CPU）。
          - 而 NET → GPU（即接收数据）时，通常还是希望数据直接到达本地 GPU，避免多余的中转和带宽损失。
          - 所以只需要补全 GPU → NET 这一个方向。
          -英文注释说明了是为了consistent with net.cc
          */
          NCCLCHECK(addInterStep(system, GPU, localGpuIndex, GPU, g, NET, n));
        }
      }
       // 8.2 如果 GPU 到 NIC 路径类型优于 PHB，检查 GDR 能力
       /*
       当 GPU 到 NIC（网卡）的路径类型优于 PHB（即不需要经过 CPU Host Bridge，通常是 PCIe 直连或多级 PCIe Switch），说明 GPU 到 NIC 的路径类型比 PHB 更优（比如 PCIe 直连），理论上可以用 GDR。
       但实际检测发现该路径不支持 GDR（GPU Direct RDMA）时，NCCL 会强制让数据流量绕行本地 CPU，即通过 CPU 作为中转节点来访问网卡。
       GDR（GPU Direct RDMA，GPU Direct Remote Direct Memory Access）是一种由 NVIDIA 推出的技术，允许网络设备（如高性能网卡）直接访问 GPU 显存（显存即 GPU 的内存），实现数据在 GPU 和网卡之间的直接传输，无需经过主机 CPU 的内存中转。 
       */
      if (gpu->paths[NET][n].type < PATH_PHB) {
        // Update path when we dont want to / can't use GPU Direct RDMA.
        int gdr;
        NCCLCHECK(ncclTopoCheckGdr(system, system->nodes[GPU].nodes[g].gpu.rank, netNode->id, 0, &gdr));
        if (gdr == 0) {
          // We cannot use GPU Direct RDMA, divert all traffic through the CPU local to the GPU
          int localCpu;
          NCCLCHECK(ncclGetLocalCpu(system, g, &localCpu));
          //这里却双向了。
          NCCLCHECK(addInterStep(system, CPU, localCpu, NET, n, GPU, g));
          NCCLCHECK(addInterStep(system, CPU, localCpu, GPU, g, NET, n));
        }
      }
    }
  }
  return ncclSuccess;
}
// 根据当前通信域（comm）和 GPU 拓扑，把不属于本通信域的 GPU 节点从系统拓扑中移除，保证后续 NCCL 通信只在本域内的 GPU 之间进行。
// 如果所有 GPU 都属于本通信域，则把所有 NET（网络）节点也移除。
ncclResult_t ncclTopoTrimSystem(struct ncclTopoSystem* system, struct ncclComm* comm) {
  ncclResult_t ret = ncclSuccess;
  int *domains;//记录每个 GPU 所属的“域”
  int64_t *ids = NULL;//：记录每个 GPU 的唯一 id。
  int myDomain = 0;//当前进程所属 GPU 的域编号。
  int ngpus = system->nodes[GPU].count;//当前系统中的 GPU 数量。
  NCCLCHECK(ncclCalloc(&domains, system->nodes[GPU].count));
  NCCLCHECKGOTO(ncclCalloc(&ids, system->nodes[GPU].count), ret, fail);
  for (int g=0; g<system->nodes[GPU].count; g++) {//外层循环遍历所有 GPU。
    struct ncclTopoNode* gpu = system->nodes[GPU].nodes+g;
    domains[g] = g;//初始时每个 GPU 自成一个域
    ids[g] = gpu->id;
    for (int p=0; p<g; p++) {//如果当前 GPU 到前面某个 GPU 的路径类型 < PATH_NET （即不是通过网络连接，而是本地互联，如 NVLink、PCIe），则把它们归为同一个域（取最小的域编号）。
      if (gpu->paths[GPU][p].type < PATH_NET) {
        domains[g] = std::min(domains[g], domains[p]);
      }
    }
    if (gpu->gpu.rank == comm->rank) myDomain = domains[g];//如果当前 GPU 的 rank 等于通信域的 rank，则记录下它的域编号
  }
//遍历所有 GPU，如果其域编号不是 myDomain ，则将其从系统拓扑中移除。
  for (int i=0; i<ngpus; i++) {
    if (domains[i] == myDomain) continue;
    struct ncclTopoNode* gpu = NULL;
    int g;
    //通过 id 在当前 GPU 节点数组中查找对应的 GPU 节点指
    //因为在前面的逻辑中， ids 数组保存了原始 GPU 节点的 id，但由于可能有节点被移除，节点数组的顺序和内容可能发生变化，所以需要通过 id 再次定位到当前系统中的对应节点。
    //这里system->nodes[GPU].count 是会变化的。
    for (g=0; g<system->nodes[GPU].count /* This one varies over the loops */; g++) {
      gpu = system->nodes[GPU].nodes+g;
      if (gpu->id == ids[i]) break; else gpu=NULL;
    }
    if (gpu == NULL) {
      WARN("Could not find id %lx", ids[i]);
      ret = ncclInternalError;
      goto fail;
    }
    //通过 id 查找对应的 GPU 节点的坐标，调用 ncclTopoRemoveNode 移除。
    NCCLCHECKGOTO(ncclTopoRemoveNode(system, GPU, g), ret, fail);
  }
  //如果剩下的 GPU 数量等于通信域的总 rank 数，说明所有 GPU 都属于本域，这时可以把所有 NET 节点（网络节点）也移除。
  //如果有 GPU 被移除，说明通信域内的 GPU 不是全部本地的，可能还需要通过网络与其他节点通信，所以不能移除网络节点。
  //对于多机多卡，system->nodes[GPU].count 是不等于nRanks的，所以网络节点不会被移除
  //
  if (system->nodes[GPU].count == comm->nRanks) {
    for (int n=system->nodes[NET].count-1; n>=0; n--)
      NCCLCHECKGOTO(ncclTopoRemoveNode(system, NET, n), ret, fail);
  }
exit:
  free(domains);
  if (ids) free(ids);
  return ret;
fail:
  goto exit;
}

void ncclTopoFree(struct ncclTopoSystem* system) {
  ncclTopoRemovePaths(system);
  free(system);
}

NCCL_PARAM(NChannelsPerNetPeer, "NCHANNELS_PER_NET_PEER", -1);

static ncclResult_t ncclTopoGetNchannels(struct ncclComm* comm, int g /*local gpu index*/, int peerRank, int* nChannels) {
  int peer;
  struct ncclTopoSystem* system = comm->topo;
  struct ncclTopoLinkList* path = NULL;
  if (ncclTopoRankToIndex(system, peerRank, &peer) == ncclSuccess) {
    // Same rank
    if (g == peer) {
      *nChannels = -1;
      return ncclSuccess;
    }
    // Local rank
    path = system->nodes[GPU].nodes[peer].paths[GPU]+g;
    if (path->type == PATH_NVL) {
      float nvlBw = ncclTopoNVLinkBw(system->nodes[GPU].nodes[g].gpu.cudaCompCap);
      *nChannels = 2*std::max(1, (int)(path->bw / nvlBw));
    } else {
      *nChannels = 2;
    }
  } else {
    // Remote rank, use network
    int nNetChannels = ncclParamNChannelsPerNetPeer();
    if (nNetChannels == -1) {
       //start from 2 channels per NIC and reduce with scale
       nNetChannels = 2;

       // check if we need to use more than one NIC, hence more than one channel
       int netCountByBw = 1, nChannelsMax = nNetChannels;
       NCCLCHECK(getLocalNetCountByBw(system, g, &netCountByBw));
       // Avoid overloading channels with 8+ operations as we loose the sync warp, hence a bit of bandwidth.
       while (nChannelsMax*comm->nRanks > comm->p2pnChannels*4 && nChannelsMax > 1) nChannelsMax /= 2;

       //allow upto channels requires to drive the NICs
       nNetChannels = std::max(netCountByBw, nChannelsMax);
    }
    *nChannels = nNetChannels;
  }
  return ncclSuccess;
}

NCCL_PARAM(MinP2pNChannels, "MIN_P2P_NCHANNELS", 1);
NCCL_PARAM(MaxP2pNChannels, "MAX_P2P_NCHANNELS", MAXCHANNELS);
extern int64_t ncclParamWorkArgsBytes();

ncclResult_t ncclTopoComputeP2pChannels(struct ncclComm* comm) {
  /* here we already honor comm->max/minCTAs for p2pnChannels. */
  if (comm->sharedRes->owner != comm) {
    comm->p2pnChannels = std::min(comm->nChannels, (int)ncclParamMaxP2pNChannels());
    comm->p2pnChannels = std::min(std::max(comm->p2pnChannels, (int)ncclParamMinP2pNChannels()), comm->sharedRes->tpP2pNChannels);
  } else {
    comm->p2pnChannels = std::min(comm->nChannels, (int)ncclParamMaxP2pNChannels());
    comm->p2pnChannels = std::max(comm->p2pnChannels, (int)ncclParamMinP2pNChannels());
  }

  int minChannels = comm->p2pnChannels;
  // We need to loop through all local GPUs to have a global picture
  for (int g=0; g<comm->topo->nodes[GPU].count; g++) {
    for (int r=0; r<comm->nRanks; r++) {
      int nChannels;
      NCCLCHECK(ncclTopoGetNchannels(comm, g, r, &nChannels));
      if (nChannels >= 0) minChannels = std::min(minChannels, nChannels);
    }
  }

  // Make nChannelsPerPeer and nChannels powers of 2. This is relied on when
  // mapping p2p peers to channels.
  comm->p2pnChannelsPerPeer = pow2Up(minChannels);
  comm->p2pnChannels = pow2Up(comm->p2pnChannels);

  comm->p2pnChannels = std::min(comm->p2pnChannels, pow2Down(ncclDevMaxChannelsForArgsBytes(ncclParamWorkArgsBytes())));
  comm->p2pnChannelsPerPeer = std::min(comm->p2pnChannelsPerPeer, comm->p2pnChannels);

  // Init channels that weren't used so far
  for (int c=comm->nChannels; c<comm->p2pnChannels; c++) NCCLCHECK(initChannel(comm, c));

  return ncclSuccess;
}

ncclResult_t ncclTopoGetNvbGpus(struct ncclTopoSystem* system, int rank, int* nranks, int** ranks) {
  int ngpus = system->nodes[GPU].count;
  NCCLCHECK(ncclCalloc(ranks, ngpus));
  int nvbGpus = 0;
  for (int g=0; g<ngpus; g++) {
    struct ncclTopoNode* gpu = system->nodes[GPU].nodes+g;
    if (gpu->gpu.rank != rank) continue;
    for (int p=0; p<ngpus; p++) {
      if (gpu->paths[GPU][p].type == PATH_NVB) {
        (*ranks)[nvbGpus++] = system->nodes[GPU].nodes[p].gpu.rank;
      }
    }
  }
  *nranks = nvbGpus;
  return ncclSuccess;
}
//获取gpu到指定类型的node的通信路径的最小路径类型
ncclResult_t ncclTopoGetGpuMinPath(struct ncclTopoSystem* system, int type, int* min) {
  int minPath = PATH_SYS;
  for (int i=0; i<system->nodes[GPU].count; i++) {
    struct ncclTopoLinkList* paths = system->nodes[GPU].nodes[i].paths[type];
    if (paths == NULL) continue;
    for (int j=0; j<system->nodes[type].count; j++) {
      if (type == GPU && i == j) continue;
      minPath = std::min(minPath, paths[j].type);
    }
  }
  *min = minPath;
  return ncclSuccess;
}
//计算gpu到指定类型的结点的通信路径的最大路径类型
ncclResult_t ncclTopoGetGpuMaxPath(struct ncclTopoSystem* system, int type, int* max) {
  int maxPath = PATH_LOC;
  for (int i=0; i<system->nodes[GPU].count; i++) {
    struct ncclTopoLinkList* paths = system->nodes[GPU].nodes[i].paths[type];
    if (paths == NULL) continue;
    for (int j=0; j<system->nodes[type].count; j++) {
      if (type == GPU && i == j) continue;
      maxPath = std::max(maxPath, paths[j].type);
    }
  }
  *max = maxPath;
  return ncclSuccess;
}

ncclResult_t ncclTopoPathAllNVLink(struct ncclTopoSystem* system, int* allNvLink) {
  int maxPath;
  NCCLCHECK(ncclTopoGetGpuMaxPath(system, GPU, &maxPath));
  *allNvLink = maxPath >= PATH_PIX ? 0 : 1;
  return ncclSuccess;
}
