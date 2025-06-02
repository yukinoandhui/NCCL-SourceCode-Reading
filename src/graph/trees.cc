/*************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "nccl.h"

#define RANK_TO_INDEX(r) (rank > root ? rank-1 : rank)

/* Btree which alternates leaves and nodes.
 * Assumes root is 0, which conveniently builds a tree on powers of two,
 * (because we have pow2-1 ranks) which lets us manipulate bits.
 * Find first non-zero bit, then :
 * Find the parent :
 *   xx01[0] -> xx10[0] (1,5,9 below) or xx00[0] if xx10[0] is out of bounds (13 below)
 *   xx11[0] -> xx10[0] (3,7,11 below)
 * 先找到最低非零位。找节点的父亲：当结尾是01的时候，就变成10。当结尾是11的时候，就把最低位变成0。特殊情况​​：若01变成...10 超出范围（如 13 的父节点应为 16，但树中无 16），则回退到 ...00（如 13 → 12）。
 * Find the children :
 *   xx10[0] -> xx01[0] (2,4,6,8,10,12) or -1 (1,3,5,7,9,11,13)
 *   xx10[0] -> xx11[0] (2,4,6,8,10) or xx101[0] (12) or xx1001[0] ... or -1 (1,3,5,7,9,11,13)
 *  特殊情况：当根节点是0的时候，其没有左子节点，右子节点是不大于节点数的最高的2次幂，例如上面14个点，0的左子节点是-1，右子节点是8。
 *  如果右子节点超出范围，则把最低位右移一位，再加上。xx10的变成xx11（就是加上lowbit>>1）发现大于范围，此时继续lowbit>>1，然后相加，变成了xx101
 * Illustration :
 * 0---------------8
 *          ______/ \______
 *         4               12
 *       /   \            /  \
 *     2       6       10     \
 *    / \     / \     /  \     \
 *   1   3   5   7   9   11    13
 */

//int rank ：当前节点（进程）的编号（从0到nranks-1），用于确定本节点在树中的位置。
//int* u ：指向上级节点（父节点）编号的指针。如果当前节点是根节点，则通常为-1。
//int* d0 ：指向第一个子节点编号的指针。如果当前节点没有子节点，则通常为-1。
//int* d1 ：指向第二个子节点编号的指针。如果当前节点没有第二个子节点，则通常为-1。
//int* parentChildType ：指向一个整数的指针，用于指示当前节点是其父节点的第一个子节点还是第二个子节点。如果当前节点是第一个子节点，则值为0；如果当前节点是第二个子节点，则值为1。
ncclResult_t ncclGetBtree(int nranks, int rank, int* u, int* d0, int* d1, int* parentChildType) {
  int up, down0, down1;
  int bit;// 找到rank的最低非零位
  for (bit=1; bit<nranks; bit<<=1) {
    if (bit & rank) break;
  }

  if (rank == 0) {// 根节点特殊处理
    *u = -1;// 没有父节点
    *d0 = -1;// 没有左子节点
    // Child rank is > 0 so it has to be our child 1, not 0.  右子节点
    *d1 = nranks > 1 ? bit >> 1 : -1;
    return ncclSuccess;
  }

  up = (rank ^ bit) | (bit << 1);
  // if smaller than the parent, we are his first child, otherwise we're his second
  if (up >= nranks) up = (rank ^ bit);
  *parentChildType = (rank < up) ? 0 : 1;
  *u = up;

  int lowbit = bit >> 1;
  // down0 is always within bounds
  down0 = lowbit == 0 ? -1 : rank-lowbit;

  down1 = lowbit == 0 ? -1 : rank+lowbit;
  // Make sure down1 is within bounds
  while (down1 >= nranks) {
    down1 = lowbit == 0 ? -1 : rank+lowbit;
    lowbit >>= 1;
  }
  *d0 = down0; *d1 = down1;

  return ncclSuccess;
}

/* Build a double binary tree. Take the previous tree for the first tree.
 * For the second tree, we use a mirror tree (if nranks is even)
 *
 * 0---------------8                   3----------------11
 *          ______/ \                 / \______
 *         4         \               /         7
 *       /   \        \             /        /   \
 *     2       6       10         1        5      9
 *    / \     / \     /  \       / \      / \    / \
 *   1   3   5   7   9   11     0   2    4   6  8   10
 *
 * or shift it by one rank (if nranks is odd).
 *
 * 0---------------8            1---------------9
 *          ______/ \______              ______/ \______
 *         4               12           5                0
 *       /   \            /           /   \            /
 *     2       6       10           3       7       11
 *    / \     / \     /  \         / \     / \     /  \
 *   1   3   5   7   9   11       2   4   6   8  10   12
 */
  /*
 为 NCCL 通信构建“双二叉树”拓扑（Double Binary Tree），即每个节点都属于两棵不同结构的二叉树
 第一棵树直接用普通二叉树（btree），第二棵树则根据进程数（nranks）奇偶性采用不同策略：
- 如果进程数为偶数，第二棵树是第一棵树的“镜像树”（mirror tree），即节点编号反转(用nranks-1-rank)。
- 如果进程数为奇数，第二棵树是第一棵树“平移一位”（shift one rank），即所有节点编号加一再取模。
 */
ncclResult_t ncclGetDtree(int nranks, int rank, int* s0, int* d0_0, int* d0_1, int* parentChildType0, int* s1, int* d1_0, int* d1_1, int* parentChildType1) {
  // First tree ... use a btree
  ncclGetBtree(nranks, rank, s0, d0_0, d0_1, parentChildType0);
  // Second tree ... mirror or shift
  if (nranks % 2 == 1) {
    // shift
    int shiftrank = (rank-1+nranks) % nranks;
    int u, d0, d1;
    ncclGetBtree(nranks, shiftrank, &u, &d0, &d1, parentChildType1);
    *s1 = u == -1 ? -1 : (u+1) % nranks;
    *d1_0 = d0 == -1 ? -1 : (d0+1) % nranks;
    *d1_1 = d1 == -1 ? -1 : (d1+1) % nranks;
  } else {
    // mirror
    int u, d0, d1;
    ncclGetBtree(nranks, nranks-1-rank, &u, &d0, &d1, parentChildType1);
    *s1 = u == -1 ? -1 : nranks-1-u;
    *d1_0 = d0 == -1 ? -1 : nranks-1-d0;
    *d1_1 = d1 == -1 ? -1 : nranks-1-d1;
  }
  return ncclSuccess;
}
