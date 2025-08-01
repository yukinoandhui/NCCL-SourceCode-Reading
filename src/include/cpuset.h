/*************************************************************************
 * Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_CPUSET_H_
#define NCCL_CPUSET_H_

// Convert local_cpus, e.g. 0003ff,f0003fff to cpu_set_t

static int hexToInt(char c) {
  int v = c - '0';
  if (v < 0) return -1;
  if (v > 9) v = 10 + c - 'a';
  if ((v < 0) || (v > 15)) return -1;
  return v;
}

#define CPU_SET_N_U32 (sizeof(cpu_set_t)/sizeof(uint32_t))
//将形如 "0003ff,f0003fff" 的十六进制字符串转换为 cpu_set_t 掩码结构
//- 0003ff ：低32位掩码，表示第0~9号CPU核可用（0x3ff = 0000 0011 1111 1111）
// 字符串中， 高位的掩码段写在后面 ，低位的掩码段写在前面
static ncclResult_t ncclStrToCpuset(const char* str, cpu_set_t* mask) {
  uint32_t cpumasks[CPU_SET_N_U32];
  int m = CPU_SET_N_U32-1;//字符串中， 高位的掩码段写在后面 ，低位的掩码段写在前面。所以这里倒着遍历。
  cpumasks[m] = 0;
  for (int o=0; o<strlen(str); o++) {
    char c = str[o];
    if (c == ',') {
      m--;
      cpumasks[m] = 0;
    } else {
      //每个字符转为16进制数（0-15），左移4位并累加到当前段。
      int v = hexToInt(c);
      if (v == -1) break;
      cpumasks[m] <<= 4;//每个十六进制字符代表4个二进制位（bits）。
      cpumasks[m] += v;
    }
  }
  // Copy cpumasks to mask
  for (int a=0; m<CPU_SET_N_U32; a++,m++) {
    memcpy(((uint32_t*)mask)+a, cpumasks+m, sizeof(uint32_t));
  }
  return ncclSuccess;
}

static ncclResult_t ncclCpusetToStr(cpu_set_t* mask, char* str) {
  int c = 0;
  uint8_t* m8 = (uint8_t*)mask;
  for (int o=sizeof(cpu_set_t)-1; o>=0; o--) {
    if (c == 0 && m8[o] == 0) continue;
    sprintf(str+c, "%02x", m8[o]);
    c+=2;
    if (o && o%4 == 0) {
      sprintf(str+c, ",");
      c++;
    }
  }
  str[c] = '\0';
  return ncclSuccess;
}

#endif
