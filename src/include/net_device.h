/*************************************************************************
 * Copyright (c) 2023-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_NET_DEVICE_H_
#define NCCL_NET_DEVICE_H_

#define NCCL_NET_DEVICE_INVALID_VERSION      0x0
#define NCCL_NET_MTU_SIZE                    4096

// Arbitrary version number - A given NCCL build will only be compatible with a single device networking plugin
// version. NCCL will check the supplied version number from net->getProperties() and compare to its internal version.
#define NCCL_NET_DEVICE_UNPACK_VERSION 0x7  
/*
  NCCL_NET_DEVICE_HOST 表示主机处理模式,在这种模式下，网络通信的处理主要由CPU完成,这是传统的网络通信模式。
  NCCL_NET_DEVICE_UNPACK 表示设备卸载模式,在这种模式下，某些网络处理功能被卸载到网络设备上执行,特别是数据包的解包操作可以直接在网卡上完成，而不需要CPU参与。
*/
typedef enum {NCCL_NET_DEVICE_HOST=0, NCCL_NET_DEVICE_UNPACK=1} ncclNetDeviceType;

typedef struct {
  ncclNetDeviceType netDeviceType; // Network offload type
  int netDeviceVersion;            // Version number for network offload
  void* handle;
  size_t size;
  int needsProxyProgress;
} ncclNetDeviceHandle_v7_t;

typedef ncclNetDeviceHandle_v7_t ncclNetDeviceHandle_v8_t;
typedef ncclNetDeviceHandle_v8_t ncclNetDeviceHandle_v9_t;
typedef ncclNetDeviceHandle_v9_t ncclNetDeviceHandle_t;

#endif
