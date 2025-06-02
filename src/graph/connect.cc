/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "device.h"
#include "graph.h"
#include "transport.h"
#include "trees.h"
#include "rings.h"
#include "topo.h"

/******************************************************************/
/********************* Internode connection ***********************/
/******************************************************************/

/*
简单来说就是设置一下channel中各个rank之间的关系
- 初始化每个通道的 ring/tree/collnet 结构体成员；
- 根据不同通信模式（ring/tree/collnet/nvls）设置每个 rank 的前后继、父子节点等信息；
- 复制通道结构体以支持多树等模式；
- 统计并去重 NVLS head 节点；
- 为后续通信连接和调度打下基础。
*/
ncclResult_t ncclTopoPreset(struct ncclComm* comm, struct ncclTopoGraph** graphs, struct ncclTopoRanks* topoRanks) {
  int rank = comm->rank;
  int localRanks = comm->topo->nodes[GPU].count;
  int nChannels = comm->nChannels;

  topoRanks->nvlsHeadNum = 0; // 初始化NVLS头节点数量为0
  for (int c=0; c<nChannels; c++) { // 遍历每个通道
    struct ncclChannel* channel = comm->channels+c;
    channel->ring.prev = channel->ring.next = -1; // 初始化ring拓扑的前后节点为-1
    channel->tree.up = -1;// 初始化tree拓扑的父节点为-1
    channel->collnetChain.up = -1;// 初始化collnet链的上游节点为-1
    for (int i=0; i<NCCL_MAX_TREE_ARITY; i++) channel->tree.down[i] = -1;// 初始化tree拓扑的所有子节点为-
    for (int i=0; i<NCCL_MAX_TREE_ARITY; i++) channel->collnetChain.down[i] = -1;// 初始化collnet链的所有下游节点为-1
    channel->collnetDirect.out = -1;// 初始化collnet直连的输出为-1
    channel->collnetDirect.headRank = -1;// 初始化collnet直连的头rank为-1
    channel->collnetDirect.nHeads = 0;
    channel->collnetDirect.shift = 0;
    for (int i=0; i<NCCL_MAX_DIRECT_ARITY+1; i++) channel->collnetDirect.heads[i] = -1;// 初始化collnet直连的所有头节点为-1
    for (int i=0; i<NCCL_MAX_DIRECT_ARITY; i++) channel->collnetDirect.up[i] = -1;
    for (int i=0; i<NCCL_MAX_DIRECT_ARITY; i++) channel->collnetDirect.down[i] = -1;
    // 获取当前通道的intra节点数组（就是当前channel对应的数组）
    int* ringIntra = graphs[NCCL_ALGO_RING]->intra+c*localRanks; 
    int* treeIntra = graphs[NCCL_ALGO_TREE]->intra+c*localRanks;
    int* collNetIntra = graphs[NCCL_ALGO_COLLNET_CHAIN]->intra+c*localRanks;

    for (int i=0; i<localRanks; i++) {
      if (ringIntra[i] == rank) { // 
        topoRanks->ringRecv[c] = ringIntra[0];// 设置当前通道的ring的头结点为第一个GPU
        topoRanks->ringSend[c] = ringIntra[localRanks-1];// 设置当前通道的ring尾节点为最后一个GPU
        topoRanks->ringPrev[c] = (i == 0) ? -1 : ringIntra[i-1];// 设置前一个节点
        topoRanks->ringNext[c] = (i == localRanks-1) ? -1 : ringIntra[i+1];// 设置下一个节点
      }
      if (treeIntra[i] == rank) {
        int parentIndex = 0;
        //这里设置为0和1，其实就是之前了解到的：每个node不一定都是local rank 0去与外界进行连接 这里的child命名有点迷惑
        //其实可以理解为child0Index和child1Index就是选谁作为代表与其他node连接。
        int child0Index = graphs[NCCL_ALGO_TREE]->pattern == NCCL_TOPO_PATTERN_TREE ? 0 : 1;
        int child1Index = graphs[NCCL_ALGO_TREE]->pattern == NCCL_TOPO_PATTERN_SPLIT_TREE ? 1 : 0;

        topoRanks->treeToParent[c] = treeIntra[parentIndex]; // 
        topoRanks->treeToChild0[c] = treeIntra[child0Index];
        topoRanks->treeToChild1[c] = treeIntra[child1Index];
        channel->tree.up         = i == 0 ? -1 : treeIntra[i-1]; //链的上一个节点
        channel->tree.down[0]    = i == localRanks-1 ? -1 : treeIntra[i+1]; //链的下一个节点
      }
      if (collNetIntra[i] == rank) {
        channel->collnetChain.up      = i == 0 ? comm->nRanks : collNetIntra[i-1];
        channel->collnetChain.down[0] = i == localRanks-1 ? -1 : collNetIntra[i+1];
      }
    }
  }
  // Duplicate channels trees
  struct ncclChannel* channel0 = comm->channels;
  struct ncclChannel* channel1 = channel0+nChannels;
  memcpy(channel1, channel0, nChannels*sizeof(struct ncclChannel));//复制

  // Get nvls heads and the number of heads. Duplicate head is not allowed.
  for (int c = 0; c < graphs[NCCL_ALGO_NVLS]->nChannels; ++c) {
    bool addHead = true;
    int* nvlsIntra = graphs[NCCL_ALGO_NVLS]->intra + c * localRanks;

    for (int dup = 0; dup < topoRanks->nvlsHeadNum; dup++) {
      if (topoRanks->nvlsHeads[dup] == nvlsIntra[0]) {
        addHead = false;
        break;
      }
    }
    if (addHead) {
      topoRanks->nvlsHeads[topoRanks->nvlsHeadNum++] = nvlsIntra[0];
    }
  }
  memcpy(comm->nvlsHeads, topoRanks->nvlsHeads, sizeof(int) * topoRanks->nvlsHeadNum);

  return ncclSuccess;
}
//建环
static ncclResult_t connectRings(struct ncclComm* comm, int* ringRecv, int* ringSend, int* ringPrev, int* ringNext) {
  int nChannels = comm->nChannels;
  int nNodes = comm->nNodes;
  for (int c=0; c<nChannels; c++) {
    int* recv = ringRecv+c*comm->nNodes;
    int* send = ringSend+c*comm->nNodes;
    int* prev = ringPrev+c*comm->nRanks;
    int* next = ringNext+c*comm->nRanks;
    for (int n=0; n<nNodes; n++) {
      int recvRank = recv[n];
      int prevSendRank = send[(n-1+nNodes)%nNodes];//上一个node的尾rank。
      prev[recvRank] = prevSendRank;//连接上一个node的尾和这个node的头
      int sendRank = send[n];
      int nextRecvRank = recv[(n+1)%nNodes];
      next[sendRank] = nextRecvRank;
    }
  }
  return ncclSuccess;
}

static ncclResult_t getIndexes(int* ranks, int* indexes, int nNodes) {
 for (int n=0; n<nNodes; n++) indexes[n] = ranks[n];
 return ncclSuccess;
}

static ncclResult_t setTreeUp(struct ncclTree* tree, int* indexes, int u) {
  if (u == -1) return ncclSuccess;
  tree->up = indexes[u];
  return ncclSuccess;
}

static ncclResult_t setTreeDown(struct ncclTree* tree, int* indexes, int d) {
  if (d == -1) return ncclSuccess;
  int x = 0;
  while (x < NCCL_MAX_TREE_ARITY && tree->down[x] >= 0) x++;
  if (x == NCCL_MAX_TREE_ARITY) {
    WARN("Internal error : tree already has %d children (%d %d %d)", x, tree->down[0], tree->down[1], tree->down[2]);
    return ncclInternalError;
  }
  tree->down[x] = indexes[d];
  return ncclSuccess;
}

static ncclResult_t connectTrees(struct ncclComm* comm, int* treeToParent, int* treeToChild0, int* treeToChild1, int* treePatterns) {
  const int nChannels = comm->nChannels, nNodes = comm->nNodes, node = comm->node;

  // Compute tree depth. Not an exact value but a good approximation in most
  // cases
  int depth = comm->nRanks/nNodes - 1 + log2i(nNodes);// 估算树的深度（近似值）

  int t0u, t0d0, t0d1, t0ChildType, t1u, t1d0, t1d1, t1ChildType;
  int* ttp, *ttc0, *ttc1; // 指向父节点和子节点的指针
  // 获取双树结构信息，具体来说就是当前node在整个集群中的父子节点（双树） t0ChildType是指当前node作为子节点的时候是左子树还是右子树
  // t0u是当前节点在树1中的父节点，t1u是当前节点在树2中的父节点，同理其他。
  NCCLCHECK(ncclGetDtree(nNodes, node, &t0u, &t0d0, &t0d1, &t0ChildType, &t1u, &t1d0, &t1d1, &t1ChildType));
  for (int c=0; c<nChannels; c++) {
     struct ncclChannel* channel0 = comm->channels+c;
     struct ncclChannel* channel1 = channel0+nChannels;// 当前通道的第二个树（双树结构）
     ttp = treeToParent+c*comm->nNodes;
     //前面preset里面的设置的treeToChild0都是每个node的loca rank 0或者1，说明就是选每个node的第一个或者第二个当做代表去和其他的node进行连接
     ttc0 = treeToChild0+c*comm->nNodes;
     ttc1 = treeToChild1+c*comm->nNodes;

     /*
     下面这些设置父子节点的，都是基于当前rank在node中的角色。如果刚好这个rank作为整个node去连接上行节点的rank，那么就需要去连接到父node的local rank0和rank1
     如果刚好这个rank作为整个node去接收下行节点的rank，那么就需要去连接到下行节点的local rank0（也就是ttp）
     */ 

     if (comm->rank == ttp[node]) {//当前机器所在的node的头rank等于comm的rank（如果当前rank是本节点的父节点）
      // 双二叉树结构。
      // 设置channel0的上行节点t0u。 
      //如果是nccl tree，则 t0ChildType==0表示当前node作为左子节点，否则说明只能作为右子节点。
      //右子节点就只能连接到父node的local rank 1.也就是分担一下流量。因为在tree的模式下，右子树默认是连接到父节点的local rank 1的，左子树则是默认连接到local rank 0。
       NCCLCHECK(setTreeUp(&channel0->tree, t0ChildType == 0 ? ttc0 : ttc1, t0u));
       // 设置channel1的上行节点。（当前node在树2中的父节点）
       NCCLCHECK(setTreeUp(&channel1->tree, t1ChildType == 0 ? ttc0 : ttc1, t1u));
     }
     if (comm->rank == ttc0[node]) {//如果当前rank是第一个子节点，那么就去设置其下行节点（他的子节点）
       NCCLCHECK(setTreeDown(&channel0->tree, ttp, t0d0));
       NCCLCHECK(setTreeDown(&channel1->tree, ttp, t1d0));
     }
     if (comm->rank == ttc1[node]) {
       NCCLCHECK(setTreeDown(&channel0->tree, ttp, t0d1));
       NCCLCHECK(setTreeDown(&channel1->tree, ttp, t1d1));
     }
     if (comm->rank == ttp[node] ||
         comm->rank == ttc0[node] ||
         comm->rank == ttc1[node]) {
       INFO(NCCL_GRAPH, "Tree %d : %d -> %d -> %d/%d/%d", c,           channel0->tree.up, comm->rank, channel0->tree.down[0], channel0->tree.down[1], channel0->tree.down[2]);
       INFO(NCCL_GRAPH, "Tree %d : %d -> %d -> %d/%d/%d", c+nChannels, channel1->tree.up, comm->rank, channel1->tree.down[0], channel1->tree.down[1], channel1->tree.down[2]);
     }
     channel0->tree.depth = channel1->tree.depth = depth;
  }
  return ncclSuccess;
}

static ncclResult_t connectCollNet(struct ncclComm* comm, struct ncclTopoGraph* collNetGraph) {
  int rank = comm->rank;
  int localRanks = comm->localRanks;
  int nHeads = 0;
  int *heads;
  NCCLCHECK(ncclCalloc(&heads, localRanks));
  // Find all head ranks
  // Head index is always 0
  for (int c=0; c<collNetGraph->nChannels; c++) {
    int* collNetIntra = collNetGraph->intra+c*localRanks;
    int head = collNetIntra[0];
    for (int h=0; h<nHeads; h++) if (heads[h] == head) head = -1;
    if (head != -1) heads[nHeads++] = collNetIntra[0];
  }
  // For all channels
  for (int c=0; c<comm->nChannels; c++) {
    struct ncclChannel* channel = comm->channels+c;
    char line[1024];
    sprintf(line, "CollNetDirect channel %d rank %d ", c, rank);
    int nDown = 0;
    for (int i=0; i<nHeads; i++) {
      if (rank == heads[i]) { // is head
        channel->collnetDirect.headRank = i; // Mark the index for deciding offset in the CUDA kernel
        channel->collnetDirect.out = comm->nRanks; // Set root of collnetDirect to id nranks
        int* collNetIntra = collNetGraph->intra+i*localRanks;
        sprintf(line+strlen(line), "down ");
        for (int r=0; r<localRanks; r++) {
          if (collNetIntra[r] == rank) continue;
          channel->collnetDirect.down[nDown++] = collNetIntra[r];  // connect to all peers
          sprintf(line+strlen(line), " %d ", collNetIntra[r]);
        }
        sprintf(line+strlen(line), "nDown %d ", nDown);
        break;
      }
    }
    // Connect to all heads
    int nUp = 0;
    sprintf(line+strlen(line), "up ");
    for (int h=0; h<nHeads; h++) {
      if (rank == heads[h]) continue;
      channel->collnetDirect.up[nUp++] = heads[h];
      sprintf(line+strlen(line), " %d ", heads[h]);
    }
    sprintf(line+strlen(line), "heads ");
    { // heads[] is the list of heads ordered in head order startubg with self
      int h0 = (channel->collnetDirect.headRank == -1) ? 0 : channel->collnetDirect.headRank;
      for (int h1=0; h1 < nHeads; h1++) {
        int h = (h0+h1)%nHeads;
        channel->collnetDirect.heads[h1] = heads[h];
        sprintf(line+strlen(line), " %d ", heads[h]);
      }
    }
    channel->collnetDirect.nHeads = nHeads;
    // nHeads should always be greater than 0.
    // coverity[divide_by_zero]
    channel->collnetDirect.shift = (rank%localRanks)%nHeads; // Shift by intraRank so that leaves don't send to same head simultaneously
    channel->collnetDirect.depth = (nUp == 0 && nDown == 0) ? 1 : 2;
    sprintf(line+strlen(line), "nUp %d nHeads %d ", nUp, nHeads);
    sprintf(line+strlen(line), "headRank %d out %d shift %d", channel->collnetDirect.headRank, channel->collnetDirect.out, channel->collnetDirect.shift);
    INFO(NCCL_GRAPH, "%s", line);
    channel->collnetChain.depth = comm->nRanks/comm->nNodes;
  }
  free(heads);
  return ncclSuccess;
}

static ncclResult_t connectNvls(struct ncclComm* comm, int* nvlsHeads, int nHeads) {
  int headRank = -1;
  if (nHeads == 0) {
    comm->nvlsChannels = 0;
    return ncclSuccess;
  }

  for (int h = 0; h < nHeads; h++) {
    if (nvlsHeads[h * comm->nNodes + comm->node] == comm->rank) headRank = h;
  }

  for (int c=0; c<comm->nChannels; c++) {
    struct ncclChannel* channel = comm->channels+c;
    channel->nvls.nHeads = nHeads;
    for (int h=0; h<nHeads; h++) channel->nvls.up[h] = comm->nRanks+1+h;
    for (int h=nHeads; h<NCCL_MAX_NVLS_ARITY; h++) channel->nvls.up[h] = -1;
    channel->nvls.down = comm->nRanks+1+headRank;
    channel->nvls.out = -1;       // NVLS+SHARP not yet implemented.
    channel->nvls.headRank = headRank;
    channel->nvls.treeUp = channel->nvls.treeDown[0] = channel->nvls.treeDown[1] = channel->nvls.treeDown[2] = -1;
    if (comm->collNetSupport && channel->nvls.headRank != -1) channel->nvls.out = comm->nRanks;
  }
  if (comm->nNodes == 1) return ncclSuccess;

  // Connect Trees
  int tree0Parent, tree0Child0, tree0Child1, tree1Parent, tree1Child0, tree1Child1;
  int pc0, pc1; // ignored
  NCCLCHECK(ncclGetDtree(comm->nNodes, comm->node,
        &tree0Parent, &tree0Child0, &tree0Child1, &pc0,
        &tree1Parent, &tree1Child0, &tree1Child1, &pc1));

  int* heads = NULL;
  int treeUp[2] = { -1, -1 };
  int treeDown0[2] = { -1, -1 };
  int treeDown1[2] = { -1, -1 };

  if (comm->node == 0) {
    for (int h=0; h<nHeads; h++) {
      char line[1024];
      sprintf(line, "NVLS Head %2d:", h);
      heads = nvlsHeads+h*comm->nNodes;
      for (int n=0; n<comm->nNodes && n<20; n++) {
        sprintf(line+strlen(line), " %2d", heads[n]);
      }
      INFO(NCCL_INIT, "%s", line);
    }
  }

  // Find the heads where I'm the head rank and retain tree up/down
  for (int h=0; h<nHeads; h++) {
    heads = nvlsHeads+h*comm->nNodes;
    if (heads[comm->node] == comm->rank) {
      treeUp[0] = tree0Parent == -1 ? -1: heads[tree0Parent];
      treeDown0[0] = tree0Child0 == -1 ? -1 : heads[tree0Child0];
      treeDown1[0] = tree0Child1 == -1 ? -1 : heads[tree0Child1];
      treeUp[1] = tree1Parent == -1 ? -1 : heads[tree1Parent];
      treeDown0[1] = tree1Child0 == -1 ? -1 : heads[tree1Child0];
      treeDown1[1] = tree1Child1 == -1 ? -1 : heads[tree1Child1];
      break;
    }
  }
  // Set prev/next in all channels (NVLS compute channels work
  // orthogonally to NVLS search channels).
  for (int c=0; c<comm->nChannels; c++) {
    struct ncclChannel* channel = comm->channels+c;
    channel->nvls.treeUp = treeUp[c%2];
    channel->nvls.treeDown[0] = channel->nvls.down;
    int ix = 1;
    if (treeDown0[c%2] != -1) channel->nvls.treeDown[ix++] = treeDown0[c%2];
    if (treeDown1[c%2] != -1) channel->nvls.treeDown[ix] = treeDown1[c%2];
  }

  struct ncclNvls* nvls0 = &comm->channels[0].nvls;
  struct ncclNvls* nvls1 = &comm->channels[1].nvls;
  INFO(NCCL_GRAPH, "NVLS Trees : %d/%d/%d->%d->%d %d/%d/%d->%d->%d",
      nvls0->treeDown[0], nvls0->treeDown[1], nvls0->treeDown[2], comm->rank, nvls0->treeUp,
      nvls1->treeDown[0], nvls1->treeDown[1], nvls1->treeDown[2], comm->rank, nvls1->treeUp);
  return ncclSuccess;
}

// Legacy naming
NCCL_PARAM(MinNrings, "MIN_NRINGS", -2);
NCCL_PARAM(MaxNrings, "MAX_NRINGS", -2);
// New naming
// 限制通道数，这个通道数也决定了用于通信的CUDA blocks 数量。
NCCL_PARAM(MinNchannels, "MIN_NCHANNELS", -2);
NCCL_PARAM(MaxNchannels, "MAX_NCHANNELS", -2);

int ncclMinNchannels() {
  int minNchannels = 0;
  if (ncclParamMinNrings() != -2) minNchannels = ncclParamMinNrings();
  if (ncclParamMinNchannels() != -2) minNchannels = ncclParamMinNchannels();
  if (minNchannels > MAXCHANNELS) {
    WARN("User asked for a minimum of %d channels, limiting to %d", minNchannels, MAXCHANNELS);
    minNchannels = MAXCHANNELS;
  }
  if (minNchannels < 0) minNchannels = 0;
  return minNchannels;
}

extern int64_t ncclParamWorkArgsBytes();
//返回NCCL允许的最大通信通道数（受环境变量NCCL_MAX_NRINGS等控制，也受内核参数空间限制）。
int ncclMaxNchannels() {
  int maxNchannels = MAXCHANNELS;
  if (ncclParamMaxNrings() != -2) maxNchannels = ncclParamMaxNrings();
  if (ncclParamMaxNchannels() != -2) maxNchannels = ncclParamMaxNchannels();
  maxNchannels = std::min(maxNchannels, ncclDevMaxChannelsForArgsBytes(ncclParamWorkArgsBytes()));
  if (maxNchannels > MAXCHANNELS) maxNchannels = MAXCHANNELS;
  if (maxNchannels < 1) {
    WARN("User asked for a maximum of %d channels, setting it to 1", maxNchannels);
    maxNchannels = 1;
  }
  return maxNchannels;
}

static int copyChannels(struct ncclComm* comm, int start, int end, int* ringPrev, int* ringNext) {
  int nranks = comm->nRanks;
  int c;
  for (c=start; c<end; c++) {
    memcpy(ringPrev+c*nranks, ringPrev+(c-start)*nranks, nranks*sizeof(int));
    memcpy(ringNext+c*nranks, ringNext+(c-start)*nranks, nranks*sizeof(int));
    memcpy(comm->channels+c, comm->channels+c-start, sizeof(struct ncclChannel));
  }
  return c;
}

void exchangeValues(int* v0, int* v1) {
  int tmp = *v1;
  *v1 = *v0;
  *v0 = tmp;
}

NCCL_PARAM(UnpackDoubleNChannels, "UNPACK_DOUBLE_NCHANNELS", 1);

ncclResult_t ncclTopoPostset(struct ncclComm* comm, int* firstRanks, int* treePatterns, struct ncclTopoRanks** allTopoRanks, int* rings, struct ncclTopoGraph** graphs, struct ncclComm* parent) {
  // Gather data from all ranks
  ncclResult_t ret = ncclSuccess;
  // 各种通信拓扑辅助数组
  int *ringRecv = NULL, *ringSend = NULL, *ringPrev = NULL, *ringNext = NULL, *treeToParent = NULL, *treeToChild0 = NULL, *treeToChild1 = NULL, *nvlsHeads = NULL;
  int nranks = comm->nRanks;
  int nNodes = comm->nNodes;
  int nChannels = comm->nChannels;
  int minHeadNum = INT_MAX;// NVLS头节点最小数量
  int shared = parent && parent->nvlsSupport  && parent->config.splitShare;// 是否为共享资源
  // 这里分配的是channelxnodes数量的二维数组大小
  NCCLCHECK(ncclCalloc(&ringRecv, nNodes*MAXCHANNELS));
  NCCLCHECKGOTO(ncclCalloc(&ringSend, nNodes*MAXCHANNELS), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&ringPrev, nranks*MAXCHANNELS), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&ringNext, nranks*MAXCHANNELS), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&treeToParent, nNodes*MAXCHANNELS), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&treeToChild0, nNodes*MAXCHANNELS), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&treeToChild1, nNodes*MAXCHANNELS), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&nvlsHeads, nNodes*MAXCHANNELS), ret, fail);

  // Alternate rings to avoid crossing rails // 交错环，避免跨 rail
  // 只对奇数节点做交换，是为了让整个 ring channel在所有节点上形成交错分布，避免所有节点的同编号channel都走同一条物理链路。
  //我的理解是：0 rank接收
  if (graphs[NCCL_ALGO_RING]->crossNic && (nChannels % 2) == 0) {
    for (int r=0; r<comm->nRanks; r++) {
      if (comm->rankToNode[r] % 2 == 1) {//只对奇数节点进行操作。
        // Exchange rings  
        for (int c=0; c<nChannels; c+=2) {
          exchangeValues(allTopoRanks[r]->ringRecv+c, allTopoRanks[r]->ringRecv+(c^1));
          exchangeValues(allTopoRanks[r]->ringSend+c, allTopoRanks[r]->ringSend+(c^1));
          exchangeValues(allTopoRanks[r]->ringPrev+c, allTopoRanks[r]->ringPrev+(c^1));
          exchangeValues(allTopoRanks[r]->ringNext+c, allTopoRanks[r]->ringNext+(c^1));
        }
      }
    }
  }
// 收集每个节点的通信信息，记录到二维数组中 具体来说就是记录每个node的channel的基本信息。
  for (int c=0; c<nChannels;c++) {
    for (int n=0; n<nNodes; n++) {
      int r = firstRanks[n];//我的理解是，随便找一个node中的rank，获取其对应的通道中的头rank、尾rank。

      ringRecv[c*nNodes+n] = allTopoRanks[r]->ringRecv[c];
      ringSend[c*nNodes+n] = allTopoRanks[r]->ringSend[c];
      //其实就是记录每个node中第一个gpu的rank信息
      treeToParent[c*nNodes+n] = allTopoRanks[r]->treeToParent[c];//local rank是0，但是这里得到的是全局的rank
      treeToChild0[c*nNodes+n] = allTopoRanks[r]->treeToChild0[c];//local rank是0或1，但是这里得到的是全局的rank
      treeToChild1[c*nNodes+n] = allTopoRanks[r]->treeToChild1[c];
    }
    //记录每个rank的prev和next信息
    for (int r=0; r<nranks; r++) {
      ringPrev[c*nranks+r] = allTopoRanks[r]->ringPrev[c];
      ringNext[c*nranks+r] = allTopoRanks[r]->ringNext[c];
    }
  }
  // 统计所有节点的最小 NVLS 头节点数
  for (int n = 0; n < nNodes; n++) {
    int r = firstRanks[n];
    if (minHeadNum > allTopoRanks[r]->nvlsHeadNum)
      minHeadNum = allTopoRanks[r]->nvlsHeadNum;
  }
  // 收集 NVLS 头节点信息
  for (int c = 0; c < minHeadNum; c++) {
    for (int n = 0; n < nNodes; n++) {
      int r = firstRanks[n];
      nvlsHeads[c * nNodes + n] = allTopoRanks[r]->nvlsHeads[c];
    }
  }

  // Connect rings and trees. This should also duplicate the channels.
  // 连接环和树结构. Tree是双树
  NCCLCHECKGOTO(connectRings(comm, ringRecv, ringSend, ringPrev, ringNext), ret, fail);
  NCCLCHECKGOTO(connectTrees(comm, treeToParent, treeToChild0, treeToChild1, treePatterns), ret, fail);

  // Duplicate ringPrev/ringNext for ncclBuildRing 复制ringPrev和ringNext
  memcpy(ringPrev+nChannels*nranks, ringPrev, nChannels*nranks*sizeof(int));
  memcpy(ringNext+nChannels*nranks, ringNext, nChannels*nranks*sizeof(int));

  // Set ring prev/next for my rank 设置环形拓扑中，每个channel中的rank的前驱和后继
  for (int c=0; c<nChannels; c++) {
    struct ncclChannel* channel0 = comm->channels+c;
    struct ncclChannel* channel1 = channel0+nChannels;
    channel0->ring.prev = channel1->ring.prev = ringPrev[c*nranks+comm->rank];
    channel0->ring.next = channel1->ring.next = ringNext[c*nranks+comm->rank];
  }

  // Duplication should be complete now 控制一下channel数量，通道复制完成，通道数加倍
  nChannels = comm->nChannels = std::min(MAXCHANNELS,nChannels*2);

  // Setup CollNet
  if (comm->collNetSupport == 1) {
    struct ncclTopoGraph* collNetChainGraph = graphs[NCCL_ALGO_COLLNET_CHAIN];
    // Add more channels to saturate intra-node bandwidth, except the 1 PPN case
    if (collNetChainGraph->bwIntra > collNetChainGraph->bwInter && comm->nRanks > comm->nNodes) {
      int collNetNchannels = std::min(MAXCHANNELS, nChannels+nChannels/2);
      // 若节点内带宽大于节点间带宽且每节点多于1个rank，则增加通道数
      //前面已经复制了很多channel，这里继续复制更多的channel
      nChannels = comm->nChannels = copyChannels(comm, nChannels, collNetNchannels, ringPrev, ringNext);
    }
    NCCLCHECKGOTO(connectCollNet(comm, graphs[NCCL_ALGO_COLLNET_DIRECT]), ret, fail);
  }

  // Use 4 compute channels per search channel to reach peak BW on <8 PPN
  // 针对 <8 PPN，提升带宽，通道数再加倍
  if (comm->minCompCap >= 90 && comm->nNodes > 1 && graphs[NCCL_ALGO_RING]->bwIntra > 45.0 && nChannels < 16) {
     nChannels = comm->nChannels = copyChannels(comm, nChannels, 2*nChannels, ringPrev, ringNext);
  }
  // unpack 网络时，节点数大于1且通道数小于16时再加倍
  // Double the number of channels when using unpack networking (greater than 1 node)
  // We won't automatically double past 16 channels, users can specify 32 if they want
  if (comm->netDeviceType == NCCL_NET_DEVICE_UNPACK && comm->nNodes > 1 && nChannels < 16 && ncclParamUnpackDoubleNChannels()) {
     nChannels = comm->nChannels = copyChannels(comm, nChannels, 2*nChannels, ringPrev, ringNext);
  }

  // Honor NCCL_MIN_NRINGS/NCCL_MAX_NRINGS. 代码会优先尊重用户通过这两个环境变量设置的通道数限制
  // We permit combining max, then min, to only use the first channels, then duplicate them.
  // 首先判断当前comm是否为共享资源的拥有者（owner），如果不是，则说明是子comm，需要限制其通道数不超过父comm。
  if (comm->sharedRes->owner != comm) {
    /* child comm #channels cannot exceed top parent #channels. */
    nChannels = comm->nChannels = std::min(std::min(std::min(ncclMaxNchannels(), nChannels), comm->config.maxCTAs), comm->sharedRes->tpNChannels);
    //取最小允许通道数和配置的最小CTA数的较大值, 再和父comm的通道数比较，不能超过父comm
    nChannels = comm->nChannels = copyChannels(comm, nChannels, std::min(std::max(ncclMinNchannels(), comm->config.minCTAs), comm->sharedRes->tpNChannels), ringPrev, ringNext);
  } else {// 如果是共享资源拥有者（即顶层comm）
    nChannels = comm->nChannels = std::min(std::min(ncclMaxNchannels(), nChannels), comm->config.maxCTAs);
    nChannels = comm->nChannels = copyChannels(comm, nChannels, std::max(ncclMinNchannels(), comm->config.minCTAs), ringPrev, ringNext);
  }

  comm->collChannels = comm->nChannels;
  //针对CUDA 12.1及以上版本，支持NVLS，进一步调整nvlsChannels，并确保不超过父comm的nvlsResources->nChannels，如果需要则再次调用copyChannels补足通道。
#if CUDART_VERSION >= 12010
  // Support maximal channel usage for aggregation
  if (shared && comm->nvlsChannels > parent->nvlsResources->nChannels) {
    comm->nvlsChannels = parent->nvlsResources->nChannels;
  }
  if (comm->nChannels < comm->nvlsChannels) {
    nChannels = comm->nChannels = copyChannels(comm, comm->nChannels, comm->nvlsChannels, ringPrev, ringNext);
  }
  NCCLCHECKGOTO(connectNvls(comm, nvlsHeads, minHeadNum), ret, fail);
  
#endif
  //再次检查如果当前comm的nChannels超过父comm的tpNChannels，则进行修正。
  if (shared && comm->nChannels > parent->sharedRes->tpNChannels) {
    nChannels = comm->nChannels = parent->sharedRes->tpNChannels;
    comm->collChannels = std::min(comm->collChannels, comm->nChannels);
  }

  // Create rings array and check all is fine
  //最后，调用ncclBuildRings函数，根据最终的nChannels数量构建ring拓扑结构。
  NCCLCHECKGOTO(ncclBuildRings(nChannels, rings, comm->rank, comm->nRanks, ringPrev, ringNext), ret, fail);

exit:
  if (ringRecv) free(ringRecv);
  if (ringSend) free(ringSend);
  if (ringPrev) free(ringPrev);
  if (ringNext) free(ringNext);
  if (treeToParent) free(treeToParent);
  if (treeToChild0) free(treeToChild0);
  if (treeToChild1) free(treeToChild1);
  if (nvlsHeads) free(nvlsHeads);
  return ret;
fail:
  goto exit;
}
