/*************************************************************************
 * Copyright (c) 2016-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "graph.h"
#include "topo.h"
#include "comm.h"
#include "nvmlwrap.h"
#include "net.h"
#include "coll_net.h"
#include "transport.h"
#include <sys/stat.h>
#include <fcntl.h>
#include "xml.h"
#include "cpuset.h"
#include "bootstrap.h"

#define BUSID_SIZE (sizeof("0000:00:00.0"))
#define BUSID_REDUCED_SIZE (sizeof("0000:00"))

const char* topoNodeTypeStr[] = { "GPU", "PCI", "NVS", "CPU", "NIC", "NET" };
const char* topoLinkTypeStr[] = { "LOC", "NVL", "",    "PCI",    "",    "",    "", "SYS", "NET" };
const char* topoPathTypeStr[] = { "LOC", "NVL", "NVB", "PIX", "PXB", "PXN", "PHB", "SYS", "NET", "DIS" };

/******************************************************************/
/******************* Graph Creation Functions *********************/
/******************************************************************/

// Get an int64 from a PCI path. For example, sys/class/pci0000:00/0000:00:02.0/0000:02:00.0/ will return 0x000002000.
ncclResult_t pciPathToInt64(char* path, int offset, int minOffset, int64_t* id) {
  char* str = path+offset;
  // Remove trailing "/"
  if (*str == '/') str--;
  // Find next /
  while (*str != '/') str--;
  str++;
  int64_t numid;
  NCCLCHECK(busIdToInt64(str, &numid));
  // Ignore subdevice because those should use the same PCI link so we want to merge nodes.
  numid -= numid & 0xf;
  *id = numid;
  return ncclSuccess;
}

static ncclResult_t findLocalCpu(struct ncclTopoNode* node, struct ncclTopoNode** cpu) {
  *cpu = NULL;
  if (node->type == CPU) {
    *cpu = node;
    return ncclSuccess;
  }
  for (int l=0; l<node->nlinks; l++) {
    // Go up the PCI tree to find the CPU. Follow only PCI switches.
    if (node->links[l].type == LINK_PCI
	&& (node->links[l].remNode->type == PCI
	    || node->links[l].remNode->type == CPU)) {
      NCCLCHECK(findLocalCpu(node->links[l].remNode, cpu));
    }
    if (*cpu != NULL) return ncclSuccess;
  }
  return ncclSuccess;
}

int interCpuBw = 0;
int cpuPciBw = 0;

static ncclResult_t ncclTopoGetInterCpuBw(struct ncclTopoNode* cpu, float* bw) {
  *bw = LOC_BW;
  if (cpu->cpu.arch == NCCL_TOPO_CPU_ARCH_POWER) {
    *bw = P9_BW;
    return ncclSuccess;
  }
  if (cpu->cpu.arch == NCCL_TOPO_CPU_ARCH_ARM) {
    *bw = ARM_BW;
    return ncclSuccess;
  }
  if (cpu->cpu.arch == NCCL_TOPO_CPU_ARCH_X86 && cpu->cpu.vendor == NCCL_TOPO_CPU_VENDOR_INTEL) {
    *bw = cpu->cpu.model == NCCL_TOPO_CPU_TYPE_SKL ? SKL_QPI_BW : QPI_BW;
  }
  if (cpu->cpu.arch == NCCL_TOPO_CPU_ARCH_X86 && cpu->cpu.vendor == NCCL_TOPO_CPU_VENDOR_AMD) {
    *bw = AMD_BW;
  }
  if (cpu->cpu.arch == NCCL_TOPO_CPU_ARCH_X86 && cpu->cpu.vendor == NCCL_TOPO_CPU_VENDOR_ZHAOXIN) {
    *bw = cpu->cpu.model ==  NCCL_TOPO_CPU_TYPE_YONGFENG ? YONGFENG_ZPI_BW : ZPI_BW;
  }
  return ncclSuccess;
}

enum ncclNvLinkDeviceType {
  ncclNvLinkDeviceUnknown,
  ncclNvLinkDeviceGpu,
  ncclNvLinkDeviceSwitch,
  ncclNvLinkDeviceBridge, // IBM/Power NVLink bridge (Device 04ea)
};

ncclResult_t ncclTopoGetNode(struct ncclTopoSystem* system, struct ncclTopoNode** node, int type, uint64_t id) {
  for (int i=0; i<system->nodes[type].count; i++) {
    if (system->nodes[type].nodes[i].id == id) {
      *node = system->nodes[type].nodes+i;
      return ncclSuccess;
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoCreateNode(struct ncclTopoSystem* system, struct ncclTopoNode** node, int type, uint64_t id) {
  if (system->nodes[type].count == NCCL_TOPO_MAX_NODES) {
    WARN("Error : tried to create too many nodes of type %d", type);
    return ncclInternalError;
  }
  struct ncclTopoNode* n = system->nodes[type].nodes+system->nodes[type].count;
  system->nodes[type].count++;
  n->type = type;
  n->id = id;
  if (type == GPU) {
    n->gpu.dev = NCCL_TOPO_UNDEF;
    n->gpu.rank = NCCL_TOPO_UNDEF;
    n->gpu.cudaCompCap = NCCL_TOPO_UNDEF;
  } else if (type == CPU) {
    n->cpu.arch = NCCL_TOPO_UNDEF;
    n->cpu.vendor = NCCL_TOPO_UNDEF;
    n->cpu.model = NCCL_TOPO_UNDEF;
  } else if (type == NET) {
    n->net.asic = 0ULL;
    n->net.port = NCCL_TOPO_UNDEF;
    n->net.bw = 0.0;
    n->net.latency = 0.0;
  }
  *node = n;
  return ncclSuccess;
}

ncclResult_t ncclTopoRemoveNode(struct ncclTopoSystem* system, int type, int index) {
  struct ncclTopoNode* delNode = system->nodes[type].nodes+index;
  for (int t=0; t<NCCL_TOPO_NODE_TYPES; t++) {
    free(delNode->paths[t]);
    for (int n=0; n<system->nodes[t].count; n++) {
      struct ncclTopoNode* node = system->nodes[t].nodes+n;
      if (node == delNode) continue;
      for (int l=0; l<node->nlinks; l++) {
        while (l<node->nlinks && node->links[l].remNode == delNode) {
          memmove(node->links+l, node->links+l+1, (node->nlinks-l-1)*sizeof(struct ncclTopoLink));
          node->nlinks--;
        }
        if (l<node->nlinks && node->links[l].remNode->type == type && node->links[l].remNode >= delNode) {
          node->links[l].remNode--;
        }
      }
    }
  }
  memmove(delNode, delNode+1, (system->nodes[type].count-index-1)*sizeof(struct ncclTopoNode));
  system->nodes[type].count--;
  return ncclSuccess;
}

ncclResult_t ncclTopoConnectNodes(struct ncclTopoNode* node, struct ncclTopoNode* remNode, int type, float bw) {
  // Aggregate links into higher bw for NVLink
  struct ncclTopoLink* link;
  for (link = node->links; link - node->links != NCCL_TOPO_MAX_LINKS && link->remNode; link++) {
    if (link->remNode == remNode && link->type == type) break;
  }
  if (link - node->links == NCCL_TOPO_MAX_LINKS) {
    WARN("Error : too many Topo links (max %d)", NCCL_TOPO_MAX_LINKS);
    return ncclInternalError;
  }
  if (link->remNode == NULL) node->nlinks++;
  link->type = type;
  link->remNode = remNode;
  link->bw += bw;

  // Sort links in BW descending order
  struct ncclTopoLink linkSave;
  memcpy(&linkSave, link, sizeof(struct ncclTopoLink));
  while (link != node->links) {
    if ((link-1)->bw >= linkSave.bw) break;
    memcpy(link, link-1, sizeof(struct ncclTopoLink));
    link--;
  }
  memcpy(link, &linkSave, sizeof(struct ncclTopoLink));
  return ncclSuccess;
}

// BCM Gen4 Switches present themselves as a two-level hierarchical switch
// even though they're supposed to sustain full BW across all ports.
// Flatten the switch as this extra level can break the search and make
// NCCL take wrong topology decisions.
int getBcmGen(uint64_t id, int level) {
  if ((id & 0xfffffffffffff000) == 0x1000c0101000a000) return 4;
  if ((id & 0xfffffffffffff000) == (0x1000c03010000000 | level*0x1000)) return 5;
  return 0;
}
ncclResult_t ncclTopoFlattenBcmSwitches(struct ncclTopoSystem* system) {
  ncclResult_t ret = ncclSuccess;
  for (int s=0; s<system->nodes[PCI].count; s++) {
    struct ncclTopoNode* pciSwitch = system->nodes[PCI].nodes+s;
    int gen = getBcmGen(pciSwitch->pci.device, 0);
    // Flatten Gen4 PEX switches in base mode
    if (gen) {
      // Find sub switches with the same device ID.
      int64_t* subSwIds;
      NCCLCHECK(ncclCalloc(&subSwIds, pciSwitch->nlinks));
      int subs = 0;
      for (int l=0; l<pciSwitch->nlinks; l++) {
        struct ncclTopoNode* sub = pciSwitch->links[l].remNode;
        // Only fuse sub switches with the same device ID.
        if (sub->type != PCI || getBcmGen(sub->pci.device, 1) != gen) continue;
        // Save sub switch for later
        subSwIds[subs++] = sub->id;
        // Remove link to that sub switch
        memmove(pciSwitch->links+l, pciSwitch->links+l+1, (pciSwitch->nlinks-l-1)*(sizeof(struct ncclTopoLink)));
        pciSwitch->nlinks--;
        // Don't increase l for the next iteration as we just shifted all links by one.
        l--;
      }

      for (int s=0; s<subs; s++) {
        // Find sub switch (system->nodes[PCI].nodes is changing every time we remove a node)
        int index;
        NCCLCHECKGOTO(ncclTopoIdToIndex(system, PCI, subSwIds[s], &index), ret, fail);
        struct ncclTopoNode* sub = system->nodes[PCI].nodes+index;
        // Connect all sub PCI devices to the parent switch
        for (int l=0; l<sub->nlinks; l++) {
          struct ncclTopoNode* remNode = sub->links[l].remNode;
          if (remNode == pciSwitch) continue;
          // Add link from parent PCI switch -> PCI device
          if (pciSwitch->nlinks == NCCL_TOPO_MAX_LINKS) {
            WARN("Error : too many Topo links (max %d)", NCCL_TOPO_MAX_LINKS);
            ret = ncclInternalError;
            goto fail;
          }
          memcpy(pciSwitch->links+pciSwitch->nlinks, sub->links+l, sizeof(struct ncclTopoLink));
          pciSwitch->nlinks++;
          // Update link from PCI device -> parent PCI switch
          for (int rl=0; rl<remNode->nlinks; rl++) {
            if (remNode->links[rl].remNode == sub) {
              remNode->links[rl].remNode = pciSwitch;
              break;
            }
          }
        }
        NCCLCHECKGOTO(ncclTopoRemoveNode(system, PCI, index), ret, fail);
      }
      // Set subdevice to 0xffff to make sure we don't merge this switch again.
      pciSwitch->pci.device |= 0xffff;
      free(subSwIds);
      // Restart, as system->nodes[PCI].nodes has changed.
      s = 0;
      continue;
fail:
      free(subSwIds);
      return ret;
    }
  }
  return ret;
}

ncclResult_t ncclTopoConnectCpus(struct ncclTopoSystem* system) {
  // And connect all CPU nodes together
  for (int n=0; n<system->nodes[CPU].count; n++) {
    struct ncclTopoNode* cpu1 = system->nodes[CPU].nodes+n;
    for (int p=0; p<system->nodes[CPU].count; p++) {
      struct ncclTopoNode* cpu2 = system->nodes[CPU].nodes+p;
      if (n == p || (NCCL_TOPO_ID_SYSTEM_ID(cpu1->id) != NCCL_TOPO_ID_SYSTEM_ID(cpu2->id))) continue;
      float bw;
      NCCLCHECK(ncclTopoGetInterCpuBw(cpu1, &bw));
      NCCLCHECK(ncclTopoConnectNodes(cpu1, cpu2, LINK_SYS, bw));
    }
  }
  return ncclSuccess;
}

static ncclResult_t ncclTopoPrintRec(struct ncclTopoNode* node, struct ncclTopoNode* prevNode, char* line, int offset) {
  if (node->type == GPU) {
    sprintf(line+offset, "%s/%lx-%lx (%d)", topoNodeTypeStr[node->type], NCCL_TOPO_ID_SYSTEM_ID(node->id), NCCL_TOPO_ID_LOCAL_ID(node->id), node->gpu.rank);
  } else if (node->type == CPU) {
    sprintf(line+offset, "%s/%lx-%lx (%d/%d/%d)", topoNodeTypeStr[node->type], NCCL_TOPO_ID_SYSTEM_ID(node->id), NCCL_TOPO_ID_LOCAL_ID(node->id), node->cpu.arch, node->cpu.vendor, node->cpu.model);
  } else if (node->type == PCI) {
    sprintf(line+offset, "%s/%lx-%lx (%lx)", topoNodeTypeStr[node->type], NCCL_TOPO_ID_SYSTEM_ID(node->id), NCCL_TOPO_ID_LOCAL_ID(node->id), node->pci.device);
  } else {
    sprintf(line+offset, "%s/%lx-%lx", topoNodeTypeStr[node->type], NCCL_TOPO_ID_SYSTEM_ID(node->id), NCCL_TOPO_ID_LOCAL_ID(node->id));
  }
  INFO(NCCL_GRAPH, "%s", line);
  for (int i=0; i<offset; i++) line[i] = ' ';

  for (int l=0; l<node->nlinks; l++) {
    struct ncclTopoLink* link = node->links+l;
    if (link->type == LINK_LOC) {
      sprintf(line+offset, "+ %s[%2.1f] - %s/%lx-%lx", topoLinkTypeStr[link->type], link->bw, topoNodeTypeStr[link->remNode->type], NCCL_TOPO_ID_SYSTEM_ID(link->remNode->id), NCCL_TOPO_ID_LOCAL_ID(link->remNode->id));
      INFO(NCCL_GRAPH, "%s", line);
    } else if (link->type != LINK_PCI || link->remNode != prevNode) {
      sprintf(line+offset, "+ %s[%2.1f] - ", topoLinkTypeStr[link->type], link->bw);
      int nextOffset = strlen(line);
      if (link->type == LINK_PCI) {
        NCCLCHECK(ncclTopoPrintRec(link->remNode, node, line, nextOffset));
      } else {
        if (link->remNode->type == NET) {
          sprintf(line+nextOffset, "%s/%lx-%lx (%d/%lx/%d/%f)", topoNodeTypeStr[link->remNode->type], NCCL_TOPO_ID_SYSTEM_ID(link->remNode->id), NCCL_TOPO_ID_LOCAL_ID(link->remNode->id), link->remNode->net.collSupport, link->remNode->net.asic, link->remNode->net.port, link->remNode->net.bw);
        } else {
          sprintf(line+nextOffset, "%s/%lx-%lx", topoNodeTypeStr[link->remNode->type], NCCL_TOPO_ID_SYSTEM_ID(link->remNode->id), NCCL_TOPO_ID_LOCAL_ID(link->remNode->id));
        }
        INFO(NCCL_GRAPH, "%s", line);
      }
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoPrint(struct ncclTopoSystem* s) {
  INFO(NCCL_GRAPH, "=== System : maxBw %2.1f totalBw %2.1f ===", s->maxBw, s->totalBw);
  char line[1024];
  for (int n=0; n<s->nodes[CPU].count; n++) NCCLCHECK(ncclTopoPrintRec(s->nodes[CPU].nodes+n, NULL, line, 0));
  INFO(NCCL_GRAPH, "==========================================");
  NCCLCHECK(ncclTopoPrintPaths(s));
  return ncclSuccess;
}

static ncclResult_t ncclTopoSort(struct ncclTopoNode* node, struct ncclTopoNode* upNode) {
  // Shift all links to have upLink as last link
  if (upNode) {
    int l=0;
    while (node->links[l].remNode != upNode) l++;
    struct ncclTopoLink upLink;
    memcpy(&upLink, node->links+l, sizeof(struct ncclTopoLink));
    while (node->links[l+1].remNode) {
      memcpy(node->links+l, node->links+l+1, sizeof(struct ncclTopoLink));
      l++;
    }
    memcpy(node->links+l, &upLink, sizeof(struct ncclTopoLink));
  }

  // Recursively sort the PCI tree
  for (int l=0; l<node->nlinks; l++) {
    struct ncclTopoLink* link = node->links+l;
    if (link->type == LINK_PCI && link->remNode != upNode) NCCLCHECK(ncclTopoSort(link->remNode, node));
  }
  return ncclSuccess;
}

// We want the graph to be organized to ease/accelerate traversal :
// 1. NVLinks (already the case)
// 2. PCI down
// 3. PCI up
// 4. SYS (already the case)
ncclResult_t ncclTopoSortSystem(struct ncclTopoSystem* system) {
  for (int n=0; n<system->nodes[CPU].count; n++) NCCLCHECK(ncclTopoSort(system->nodes[CPU].nodes+n, NULL));
  return ncclSuccess;
}

ncclResult_t ncclTopoAddNet(struct ncclXmlNode* xmlNet, struct ncclTopoSystem* system, struct ncclTopoNode* nic, int systemId) {
  int dev;
  NCCLCHECK(xmlGetAttrInt(xmlNet, "dev", &dev));

  struct ncclTopoNode* net;
  NCCLCHECK(ncclTopoCreateNode(system, &net, NET, NCCL_TOPO_ID(systemId, dev)));
  net->net.dev = dev;
  const char* str;
  NCCLCHECK(xmlGetAttr(xmlNet, "guid", &str));
  if (str) sscanf(str, "0x%lx", &net->net.asic);
  else net->net.asic = dev;

  ncclDebugNoWarn = NCCL_GRAPH;
  int mbps;
  NCCLCHECK(xmlGetAttrIntDefault(xmlNet, "speed", &mbps, 0));
  if (mbps <= 0) mbps = 10000; // Some NICs define speed = -1
  net->net.bw = mbps / 8000.0;
  if (xmlGetAttrFloat(xmlNet, "latency", &net->net.latency) != ncclSuccess) net->net.latency = 0;
  NCCLCHECK(xmlGetAttrIntDefault(xmlNet, "port", &net->net.port, 0));
  NCCLCHECK(xmlGetAttrIntDefault(xmlNet, "gdr", &net->net.gdrSupport, 0));
  NCCLCHECK(xmlGetAttrIntDefault(xmlNet, "maxconn", &net->net.maxChannels, MAXCHANNELS));
  NCCLCHECK(xmlGetAttrIntDefault(xmlNet, "coll", &net->net.collSupport, 0));
  ncclDebugNoWarn = 0;

  NCCLCHECK(ncclTopoConnectNodes(nic, net, LINK_NET, net->net.bw));
  NCCLCHECK(ncclTopoConnectNodes(net, nic, LINK_NET, net->net.bw));
  return ncclSuccess;
}

ncclResult_t ncclTopoAddNic(struct ncclXmlNode* xmlNic, struct ncclTopoSystem* system, struct ncclTopoNode* nic, int systemId) {
  for (int s=0; s<xmlNic->nSubs; s++) {
    struct ncclXmlNode* xmlNet = xmlNic->subs[s];
    if (strcmp(xmlNet->name, "net") != 0) continue;
    int index;
    NCCLCHECK(xmlGetAttrIndex(xmlNet, "dev", &index));
    // This means that the "dev" attribute wasn't set on this net xml node. That means it should not be added to the system topology graph
    if (index == -1) continue;
    NCCLCHECK(ncclTopoAddNet(xmlNet, system, nic, systemId));
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoAddGpu(struct ncclXmlNode* xmlGpu, struct ncclTopoSystem* system, struct ncclTopoNode* gpu) {
  NCCLCHECK(xmlGetAttrInt(xmlGpu, "sm", &gpu->gpu.cudaCompCap));
  NCCLCHECK(xmlGetAttrInt(xmlGpu, "rank", &gpu->gpu.rank));
  NCCLCHECK(xmlGetAttrInt(xmlGpu, "dev", &gpu->gpu.dev));
  NCCLCHECK(xmlGetAttrInt(xmlGpu, "gdr", &gpu->gpu.gdrSupport));
  // Do not go any further, nvlinks will be added in a second pass
  return ncclSuccess;
}

struct kvDict kvDictPciClass[] = { { "0x060400", PCI }, { "0x068000", NVS }, { "0x068001", CPU }, { "0x03", GPU }, { "0x02", NIC }, { NULL, PCI /* Default fallback value */ } };
struct kvDict kvDictPciGen[] = {
  { "2.5 GT/s", 15 }, { "5 GT/s", 30 }, { "8 GT/s", 60 }, { "16 GT/s", 120 }, { "32 GT/s", 240 }, /* Kernel 5.6 and earlier */
  { "2.5 GT/s PCIe", 15 }, { "5.0 GT/s PCIe", 30 }, { "8.0 GT/s PCIe", 60 }, { "16.0 GT/s PCIe", 120 }, { "32.0 GT/s PCIe", 240 }, { "64.0 GT/s PCIe", 480 },
  { NULL, 60 /* Default fallback */ } }; // x100 Mbps per lane
ncclResult_t ncclTopoAddPci(struct ncclXmlNode* xmlPci, struct ncclTopoSystem* system, struct ncclTopoNode* parent, int systemId, int numaId) {
  const char* str;

  int type;
  NCCLCHECK(xmlGetAttrStr(xmlPci, "class", &str));
  NCCLCHECK(kvConvertToInt(str, &type, kvDictPciClass));

  int64_t busId;
  NCCLCHECK(xmlGetAttrStr(xmlPci, "busid", &str));
  NCCLCHECK(busIdToInt64(str, &busId));

  struct ncclTopoNode* node = NULL;
  struct ncclXmlNode* xmlGpu = NULL;
  NCCLCHECK(xmlGetSub(xmlPci, "gpu", &xmlGpu));
  if (xmlGpu != NULL) {
    type = GPU;
    int index;
    NCCLCHECK(xmlGetAttrIndex(xmlGpu, "rank", &index));
    if (index == -1) return ncclSuccess;
    NCCLCHECK(ncclTopoCreateNode(system, &node, type, NCCL_TOPO_ID(systemId, busId)));
    NCCLCHECK(ncclTopoAddGpu(xmlGpu, system, node));
  }
  struct ncclXmlNode* xmlNic = NULL;
  NCCLCHECK(xmlGetSub(xmlPci, "nic", &xmlNic));
  if (xmlNic != NULL) {
    type = NIC;
    // Ignore sub device ID and merge multi-port NICs into one PCI device.
    struct ncclTopoNode* nicNode = NULL;
    int64_t localNicId = NCCL_TOPO_LOCAL_NIC_ID(numaId, busId);
    int64_t id = NCCL_TOPO_ID(systemId, localNicId);
    NCCLCHECK(ncclTopoGetNode(system, &nicNode, type, id));
    if (nicNode == NULL) {
      NCCLCHECK(ncclTopoCreateNode(system, &nicNode, type, id));
      node = nicNode; // Connect it to parent later on
    }
    NCCLCHECK(ncclTopoAddNic(xmlNic, system, nicNode, systemId));
  } else if (type == PCI) {
    NCCLCHECK(ncclTopoCreateNode(system, &node, type, NCCL_TOPO_ID(systemId, busId)));
    NCCLCHECK(xmlGetAttr(xmlPci, "vendor", &str));
    if (str) node->pci.device += strtol(str, NULL, 0) << 48;
    NCCLCHECK(xmlGetAttr(xmlPci, "device", &str));
    if (str) node->pci.device += strtol(str, NULL, 0) << 32;
    NCCLCHECK(xmlGetAttr(xmlPci, "subsystem_vendor", &str));
    if (str) node->pci.device += strtol(str, NULL, 0) << 16;
    NCCLCHECK(xmlGetAttr(xmlPci, "subsystem_device", &str));
    if (str) node->pci.device += strtol(str, NULL, 0);

    for (int s=0; s<xmlPci->nSubs; s++) {
      struct ncclXmlNode* xmlSubPci = xmlPci->subs[s];
      if (strcmp(xmlSubPci->name, "pcilink") != 0) { // PCI links will be added later
        NCCLCHECK(ncclTopoAddPci(xmlSubPci, system, node, systemId, numaId));
      }
    }
  }

  if (node) {
    int width, speed;
    NCCLCHECK(xmlGetAttrInt(xmlPci, "link_width", &width));
    NCCLCHECK(xmlGetAttrStr(xmlPci, "link_speed", &str));

    // Manage cases where speed was not indicated in /sys
    if (width == 0) width = 16;
    NCCLCHECK(kvConvertToInt(str, &speed, kvDictPciGen)); // Values in 100Mbps, per lane (we want GB/s in the end)

    NCCLCHECK(ncclTopoConnectNodes(node, parent, LINK_PCI, width*speed/80.0));
    NCCLCHECK(ncclTopoConnectNodes(parent, node, LINK_PCI, width*speed/80.0));
  }
  return ncclSuccess;
}

struct kvDict kvDictCpuArch[] = { { "x86_64", NCCL_TOPO_CPU_ARCH_X86 }, { "arm64", NCCL_TOPO_CPU_ARCH_ARM }, { "ppc64", NCCL_TOPO_CPU_ARCH_POWER }, { NULL, 0 } };
struct kvDict kvDictCpuVendor[] = { { "GenuineIntel", NCCL_TOPO_CPU_VENDOR_INTEL }, { "AuthenticAMD", NCCL_TOPO_CPU_VENDOR_AMD }, { "CentaurHauls", NCCL_TOPO_CPU_VENDOR_ZHAOXIN }, { "  Shanghai  ", NCCL_TOPO_CPU_VENDOR_ZHAOXIN }, { NULL, 0 } };

ncclResult_t ncclGetSystemId(struct ncclTopoSystem* system, struct ncclXmlNode* xmlCpu, int* systemIdPtr) {
  const char* hostHashStr;
  NCCLCHECK(xmlGetAttr(xmlCpu, "host_hash", &hostHashStr));
  uint64_t hostHash = hostHashStr ? strtoull(hostHashStr, NULL, 16) : 0;
  int systemId;
  for (systemId=0; systemId<system->nHosts; systemId++) if (system->hostHashes[systemId] == hostHash) break;
  if (systemId == system->nHosts) system->hostHashes[system->nHosts++] = hostHash;
  *systemIdPtr = systemId;
  return ncclSuccess;
}


ncclResult_t ncclTopoAddCpu(struct ncclXmlNode* xmlCpu, struct ncclTopoSystem* system) {
  int numaId;
  NCCLCHECK(xmlGetAttrInt(xmlCpu, "numaid", &numaId));
  int systemId;
  NCCLCHECK(ncclGetSystemId(system, xmlCpu, &systemId));
  struct ncclTopoNode* cpu;
  NCCLCHECK(ncclTopoCreateNode(system, &cpu, CPU, NCCL_TOPO_ID(systemId, numaId)));
  const char* str;
  NCCLCHECK(xmlGetAttr(xmlCpu, "affinity", &str));
  if (str != NULL) {
    NCCLCHECK(ncclStrToCpuset(str, &cpu->cpu.affinity));
  }

  NCCLCHECK(xmlGetAttrStr(xmlCpu, "arch", &str));
  NCCLCHECK(kvConvertToInt(str, &cpu->cpu.arch, kvDictCpuArch));
  if (cpu->cpu.arch == NCCL_TOPO_CPU_ARCH_X86) {
    NCCLCHECK(xmlGetAttrStr(xmlCpu, "vendor", &str));
    NCCLCHECK(kvConvertToInt(str, &cpu->cpu.vendor, kvDictCpuVendor));
    if (cpu->cpu.vendor == NCCL_TOPO_CPU_VENDOR_INTEL) {
      int familyId, modelId;
      NCCLCHECK(xmlGetAttrInt(xmlCpu, "familyid", &familyId));
      NCCLCHECK(xmlGetAttrInt(xmlCpu, "modelid", &modelId));
      cpu->cpu.model = (familyId == 6 && modelId >= 0x55) ? NCCL_TOPO_CPU_TYPE_SKL : NCCL_TOPO_CPU_INTEL_BDW;
    } else if (cpu->cpu.vendor == NCCL_TOPO_CPU_VENDOR_ZHAOXIN) {
      int familyId, modelId;
      NCCLCHECK(xmlGetAttrInt(xmlCpu, "familyid", &familyId));
      NCCLCHECK(xmlGetAttrInt(xmlCpu, "modelid", &modelId));
      if (familyId == 7 && modelId == 0x5B) cpu->cpu.model = NCCL_TOPO_CPU_TYPE_YONGFENG;
    }
  }
  for (int s=0; s<xmlCpu->nSubs; s++) {
    struct ncclXmlNode* node = xmlCpu->subs[s];
    if (strcmp(node->name, "pci") == 0) NCCLCHECK(ncclTopoAddPci(node, system, cpu, systemId, numaId));
    if (strcmp(node->name, "nic") == 0) {
      struct ncclTopoNode* nic = NULL;
      int64_t localNicId = NCCL_TOPO_LOCAL_NIC_ID(numaId, 0);
      int64_t id = NCCL_TOPO_ID(systemId, localNicId);
      NCCLCHECK(ncclTopoGetNode(system, &nic, NIC, id));
      if (nic == NULL) {
        NCCLCHECK(ncclTopoCreateNode(system, &nic, NIC, id));
        NCCLCHECK(ncclTopoConnectNodes(cpu, nic, LINK_PCI, LOC_BW));
        NCCLCHECK(ncclTopoConnectNodes(nic, cpu, LINK_PCI, LOC_BW));
      }
      NCCLCHECK(ncclTopoAddNic(node, system, nic, systemId));
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoAddNvLinks(struct ncclXmlNode* node, struct ncclTopoSystem* system, const char* parentBusId, int systemId) {
  if (strcmp(node->name, "nvlink") == 0) {
    struct ncclTopoNode* gpu = NULL;
    int64_t pBusId;
    NCCLCHECK(busIdToInt64(parentBusId, &pBusId));
    pBusId = NCCL_TOPO_ID(systemId, pBusId);
    NCCLCHECK(ncclTopoGetNode(system, &gpu, GPU, pBusId));
    if (gpu == NULL) {
      WARN("Add NVLink error : could not find GPU %lx", pBusId);
      return ncclInternalError;
    }
    int count;
    NCCLCHECK(xmlGetAttrInt(node, "count", &count));
    const char* targetClass;
    NCCLCHECK(xmlGetAttrStr(node, "tclass", &targetClass));
    int targetType;
    NCCLCHECK(kvConvertToInt(targetClass, &targetType, kvDictPciClass));
    struct ncclTopoNode* remote = NULL;
    if (targetType == GPU) {
      // NVL P2P connection to another GPU
      const char* target;
      NCCLCHECK(xmlGetAttrStr(node, "target", &target));
      int64_t busId;
      NCCLCHECK(busIdToInt64(target, &busId));
      NCCLCHECK(ncclTopoGetNode(system, &remote, GPU, NCCL_TOPO_ID(systemId, busId)));
    } else if (targetType == CPU) {
      // NVL connection to the local CPU
      NCCLCHECK(findLocalCpu(gpu, &remote));
    } else {
      if (system->nodes[NVS].count == 0) {
        NCCLCHECK(ncclTopoCreateNode(system, &remote, NVS, 0));
      } else {
        remote = system->nodes[NVS].nodes;
      }
    }
    if (remote) {
      float nvlBw = ncclTopoNVLinkBw(gpu->gpu.cudaCompCap);
      NCCLCHECK(ncclTopoConnectNodes(gpu, remote, LINK_NVL, count*nvlBw));
      if (remote->type != GPU) {
        NCCLCHECK(ncclTopoConnectNodes(remote, gpu, LINK_NVL, count*nvlBw));
      }
    }
  } else {
    if (strcmp(node->name, "cpu") == 0) {
      NCCLCHECK(ncclGetSystemId(system, node, &systemId));
    }
    const char* busId;
    NCCLCHECK(xmlGetAttr(node, "busid", &busId));
    for (int s=0; s<node->nSubs; s++) {
      NCCLCHECK(ncclTopoAddNvLinks(node->subs[s], system, busId ? busId : parentBusId, systemId));
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoAddPciLinks(struct ncclXmlNode* node, struct ncclTopoSystem* system, const char* parentBusId, int systemId) {
  if (strcmp(node->name, "pcilink") == 0) {
    struct ncclTopoNode* pci = NULL;
    int64_t pBusId;
    NCCLCHECK(busIdToInt64(parentBusId, &pBusId));
    pBusId = NCCL_TOPO_ID(systemId, pBusId);
    NCCLCHECK(ncclTopoGetNode(system, &pci, PCI, pBusId));
    if (pci == NULL) {
      WARN("Add PCI Link error : could not find PCI SW %lx", pBusId);
      return ncclInternalError;
    }
    struct ncclTopoNode* remote = NULL;
    const char* target;
    NCCLCHECK(xmlGetAttrStr(node, "target", &target));
    int64_t busId;
    NCCLCHECK(busIdToInt64(target, &busId));
    NCCLCHECK(ncclTopoGetNode(system, &remote, PCI, NCCL_TOPO_ID(systemId, busId)));
    if (remote) NCCLCHECK(ncclTopoConnectNodes(pci, remote, LINK_LOC, LOC_BW));
  } else {
    if (strcmp(node->name, "cpu") == 0) {
      NCCLCHECK(ncclGetSystemId(system, node, &systemId));
    }
    const char* busId;
    NCCLCHECK(xmlGetAttr(node, "busid", &busId));
    for (int s=0; s<node->nSubs; s++) {
      NCCLCHECK(ncclTopoAddPciLinks(node->subs[s], system, busId ? busId : parentBusId, systemId));
    }
  }
  return ncclSuccess;
}


ncclResult_t ncclTopoAddC2c(struct ncclXmlNode* node, struct ncclTopoSystem* system, const char* parentBusId, int systemId) {
  if (strcmp(node->name, "c2c") == 0) {
    struct ncclTopoNode* gpu = NULL;
    int64_t pBusId;
    NCCLCHECK(busIdToInt64(parentBusId, &pBusId));
    pBusId = NCCL_TOPO_ID(systemId, pBusId);
    NCCLCHECK(ncclTopoGetNode(system, &gpu, GPU, pBusId));
    if (gpu == NULL) {
      WARN("Add NVLink error : could not find GPU %lx", pBusId);
      return ncclInternalError;
    }
    int count = 0;
    NCCLCHECK(xmlGetAttrInt(node, "count", &count));
    int bw = 0;
    NCCLCHECK(xmlGetAttrInt(node, "bw", &bw));
    double c2cBw = (bw*count)/1000.0;
    struct ncclTopoNode* cpu = NULL;
    NCCLCHECK(findLocalCpu(gpu, &cpu));
    if (cpu == NULL) return ncclSuccess;
    NCCLCHECK(ncclTopoConnectNodes(gpu, cpu, LINK_NVL, c2cBw));
    NCCLCHECK(ncclTopoConnectNodes(cpu, gpu, LINK_NVL, c2cBw));
  } else {
    if (strcmp(node->name, "cpu") == 0) {
      NCCLCHECK(ncclGetSystemId(system, node, &systemId));
    }
    const char* busId;
    NCCLCHECK(xmlGetAttr(node, "busid", &busId));
    for (int s=0; s<node->nSubs; s++) {
      NCCLCHECK(ncclTopoAddC2c(node->subs[s], system, busId ? busId : parentBusId, systemId));
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoGetSystemFromXml(struct ncclXml* xml, struct ncclTopoSystem** topoSystem, const uint64_t localHostHash) {
  NCCLCHECK(ncclCalloc(topoSystem, 1));
  struct ncclTopoSystem* system = *topoSystem;
  struct ncclXmlNode* topNode;
  NCCLCHECK(xmlFindTag(xml, "system", &topNode));
  for (int s=0; s<topNode->nSubs; s++) {
    struct ncclXmlNode* node = topNode->subs[s];
    if (strcmp(node->name, "cpu") == 0) NCCLCHECK(ncclTopoAddCpu(node, *topoSystem));
  }
  for (int systemId=0; systemId<system->nHosts; systemId++) if (system->hostHashes[systemId] == localHostHash) system->systemId = systemId;

  NCCLCHECK(ncclTopoAddNvLinks(topNode, *topoSystem, NULL, 0));
  NCCLCHECK(ncclTopoAddC2c(topNode, *topoSystem, NULL, 0));
  NCCLCHECK(ncclTopoAddPciLinks(topNode, *topoSystem, NULL, 0));

  NCCLCHECK(ncclTopoFlattenBcmSwitches(*topoSystem));
  NCCLCHECK(ncclTopoConnectCpus(*topoSystem));
  NCCLCHECK(ncclTopoSortSystem(*topoSystem));

  return ncclSuccess;
}

NCCL_PARAM(TopoDumpFileRank, "TOPO_DUMP_FILE_RANK", 0);

// Only set values if not already set
static ncclResult_t xmlInitAttrInt(struct ncclXmlNode* node, const char* attrName, const int value) {
  int index;
  NCCLCHECK(xmlGetAttrIndex(node, attrName, &index));
  if (index == -1) {
    index = node->nAttrs++;
    strncpy(node->attrs[index].key, attrName, MAX_STR_LEN);
    snprintf(node->attrs[index].value, MAX_STR_LEN, "%d", value);
  }
  return ncclSuccess;
}
static ncclResult_t xmlInitAttrUint64(struct ncclXmlNode* node, const char* attrName, const uint64_t value) {
  int index;
  NCCLCHECK(xmlGetAttrIndex(node, attrName, &index));
  if (index == -1) {
    index = node->nAttrs++;
    strncpy(node->attrs[index].key, attrName, MAX_STR_LEN);
    snprintf(node->attrs[index].value, MAX_STR_LEN, "0x%lx", value);
  }
  return ncclSuccess;
}
static ncclResult_t xmlInitAttrFloat(struct ncclXmlNode* node, const char* attrName, const float value) {
  int index;
  NCCLCHECK(xmlGetAttrIndex(node, attrName, &index));
  if (index == -1) {
    index = node->nAttrs++;
    strncpy(node->attrs[index].key, attrName, MAX_STR_LEN);
    snprintf(node->attrs[index].value, MAX_STR_LEN, "%f", value);
  }
  return ncclSuccess;
}
//刷新 Broadcom（BCM） PCIe 交换机的 P2P（Peer-to-Peer）拓扑信息 ，以确保 NCCL 能够获取到最新的 PCIe 拓扑结构
ncclResult_t ncclTopoRefreshBcmP2pLinks(void) {
  //refresh the switch topology by reading the link below
  //通过读取的这个动作，触发内核或驱动刷新 PCIe 交换机的拓扑信息
  //主要针对 Broadcom（BCM）等 PCIe 交换机，因为这些交换机的拓扑可能会动态变化（比如设备热插拔、链路变化等）。
  //这是 Linux 下常见的“驱动接口”方式：驱动开发者会在 /sys 下暴露一些特殊文件，用户空间只要读/写这些文件，驱动就会执行相应的操作（比如刷新硬件状态、重建拓扑等）
  FILE *fp = fopen("/sys/kernel/pci_switch_link/refresh_switch_toplogy", "r");
  if (fp != NULL) {
    int tmp;
    size_t r = fread(&tmp, sizeof(tmp), 1, fp);
    if (r != 1)
      INFO(NCCL_GRAPH, "Failed to read refresh_switch_toplogy");
    fclose(fp);
  }
  return ncclSuccess;
}

// This is just checking for direct descendence
int ncclTopoCheckPix(ncclXmlNode* common, ncclXmlNode** nodes, int nNodes) {
  const char* tempBusId;
  // If the common parent isn't a pci switch, then this isn't PIX
  NCCLCHECK(xmlGetAttrStr(common, "busid", &tempBusId));
  if (tempBusId == NULL) return 0;
  TRACE(NCCL_GRAPH, "Checking pix for busid=%s", tempBusId);

  // All the nodes must have a "nic" which is a parent, and then a pci node (busid) which must be a child of the "common"
  for (int i = 0; i < nNodes; i++) {
    ncclXmlNode* node = nodes[i];
    if (strcmp(node->name, "net") == 0) {
      node = node->parent;
      if (node == NULL) return 0;
      if (strcmp(node->name, "nic") == 0) {
        node = node->parent;
        if (node == NULL) return 0;
        // All nodes must descend from the same first level pci switch
        if (strcmp(node->name, "pci") == 0) {
          TRACE(NCCL_GRAPH, "Comparing parent of node=%p to common=%p", node->parent, common);
          if (node->parent != common) return 0;
        }
      }
    }
  }

  return 1;
}

#define NCCL_TOPO_XML_DEPTH_MAX 256
typedef struct xmlNodeStack {
  ncclXmlNode* elems[NCCL_TOPO_XML_DEPTH_MAX];
  int tail;

  ncclXmlNode* top() {
    if (!empty()) {
      return elems[tail - 1];
    } else {
      return NULL;
    }
  }

  ncclXmlNode* pop() {
    ncclXmlNode* node = top();
    if (node) {
      tail--;
    }
    return node;
  }

  void push(ncclXmlNode* node) {
    if (tail < NCCL_TOPO_XML_DEPTH_MAX) {
      elems[tail++] = node;
    }
  }

  bool empty() {
    return tail == 0;
  }

} xmlNodeStack;

// 1. Find the common parent xmlNode between the given set of nodes
/*
用于判断一组节点（通常是物理网卡节点）在 NCCL 拓扑 XML 树中的“共同父节点”以及它们之间的路径类型（如是否同一端口、同一PCI树、跨NUMA等），
并返回路径类型和共同父节点指针。
多端口（multi-port）判断是为了支持多端口网卡或 SR-IOV 场景的特殊合并需求。
假设一台服务器上有一块物理网卡（NIC），但这块网卡有两个物理端口（比如 Infiniband HCA 的 port 1 和 port 2），并且这两个端口都被操作系统识别为独立的网络接口（如 mlx5_0/port1 和 mlx5_0/port2）。

在 NCCL 拓扑 XML 中，通常会有如下结构（伪代码）：

<system>
  ...
  <nic>
    <net name="mlx5_0_port1" dev="0" guid="0x1234" port="1"/>
    <net name="mlx5_0_port2" dev="1" guid="0x1234" port="2"/>
  </nic>
  ...
</system>

### 合并前
- mlx5_0_port1 和 mlx5_0_port2 是两个独立的 net 节点。
- 但它们的父节点都是同一个 <nic> ，且 guid 相同，只是 port 不同。
### 多端口判断的作用
在 NCCL 拓扑合并时，会判断这些 net 节点的父节点（即 <nic> ）是否相同，或者它们的 busid 除了最后一位（port号）外是否一致。

- 如果一致，就认为它们属于同一块物理网卡的不同端口，可以合并为一个虚拟网卡（VNIC）。
### 合并后
NCCL 会把这两个端口合并为一个虚拟网卡节点，后续通信调度时可以统一管理和优化。


在 ncclTopoGetPath 里，会有类似如下逻辑：

- 如果多个 net 节点的父节点 busid 除了最后一位都相同（即同一物理卡的不同端口），则路径类型为 PATH_PORT ，允许合并。
*/
ncclResult_t ncclTopoGetPath(ncclXmlNode** nodes, int nNodes, int* path, ncclXmlNode** parent) {
  // Track a stack of parents per-net node being merged
  xmlNodeStack* parents;
  NCCLCHECK(ncclCalloc(&parents, nNodes));
  // Find the common parent
  ncclXmlNode* common = NULL;

  if (nNodes == 1) {
    common = nodes[0];
    *path = PATH_LOC;
    goto out;
  }

  for (int i = 0; i < nNodes; i++) {
    ncclXmlNode* temp;
    temp = nodes[i];
    while (temp) {
      parents[i].push(temp);
      temp = strcmp(temp->name, "system") == 0 ? NULL : temp->parent;
    }
  }

  common = NULL;
  int c;
  c = 1;
  while (c && !parents[0].empty()) {
    ncclXmlNode* temp = parents[0].top();
    for (int i = 1; i < nNodes; i++) {
      if (!parents[i].empty()) {
        c &= (temp == parents[i].top());
      } else {
        c = 0;
        break;
      }
    }

    if (c) {
      common = temp;
      if (common == NULL) TRACE(NCCL_GRAPH, "COMMON IS NULL");
      for (int i = 0; i < nNodes; i++) {
        parents[i].pop();
      }
    // Check multi-port while we still have the mismatched parents
    // For multi-port to be true, all parents (peers) must have the busId attribute with all but the last character matching
    } else {
      int multiPort = 1;
      const char* tempBusId;

      NCCLCHECK(xmlGetAttr(temp, "busid", &tempBusId));
      if (tempBusId) {
        for (int i = 1; i < nNodes; i++) {
          if (!parents[i].empty()) {
            const char* busId;
            NCCLCHECK(xmlGetAttr(parents[i].top(), "busid", &busId));
            if (busId) {
              if (strlen(busId) != strlen(tempBusId)) {
                multiPort = 0;
                break;
              }
              if (strncmp(busId, tempBusId, strlen(busId)-1) != 0) {
                multiPort = 0;
                break;
              }
            } else {
              multiPort = 0;
              break;
            }
          }
        }
      } else {
        multiPort = 0;
      }

      if (multiPort) {
        *path = PATH_PORT;
        goto out;
      }
    }
  }

  if (common == NULL) {
    *path = PATH_DIS;
  } else if (strcmp(common->name,"system") == 0) {
    *path = PATH_SYS;
  } else if (strcmp(common->name, "cpu") == 0) {
    *path = PATH_PHB;
  } else if (strcmp(common->name, "nic") == 0) {
    *path = PATH_PORT;
  } else if (strcmp(common->name, "net") == 0) {
    *path = PATH_PORT;
  } else if (ncclTopoCheckPix(common, nodes, nNodes)) {
    *path = PATH_PIX;
  } else {
    *path = PATH_PXB;
  }

out:
  *parent = common;
  free(parents);
  return ncclSuccess;
}

ncclResult_t ncclTopoMakeUniqueBusId(struct ncclXml* xml, char* busId, struct ncclXmlNode** pciNode, struct ncclXmlNode* parent) {
  int i = 0;
  int64_t rBusId;
  NCCLCHECK(busIdToInt64(busId, &rBusId));
  // Try to find an unused busid - NCCL expects leaf busid to be unique
  while (i < 100) {
    rBusId++;
    TRACE(NCCL_GRAPH, "Trying to make new busId %lx", rBusId);
    int64ToBusId(rBusId, busId);
    struct ncclXmlNode* temp = NULL;
    NCCLCHECK(xmlFindTagKv(xml, "pci", &temp, "busid", busId));
    if (temp == NULL) {
      NCCLCHECK(xmlAddNode(xml, parent, "pci", pciNode));
      NCCLCHECK(xmlSetAttr(*pciNode, "busid", busId));
      TRACE(NCCL_GRAPH, "Made new busId %lx", rBusId);
      return ncclSuccess;
    }
    TRACE(NCCL_GRAPH, "Conflicting busId %lx", rBusId);
    i++;
  }

  WARN("TOPO/NET : Couldn't generate unique busId after %d tries", i);
  return ncclInternalError;
}

ncclResult_t ncclTopoMakePciParent(struct ncclXml* xml, struct ncclXmlNode** parent, struct ncclXmlNode* physNetNode) {
  struct ncclXmlNode* newBusId = NULL;
  struct ncclXmlNode* pci = physNetNode->parent;
  if (pci) {
    pci = pci->parent;
    if (pci) {
      if (strcmp(pci->name, "pci") == 0) {
        char busId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
        memset(busId, 0, sizeof(busId));
        const char* originalBusId;
        // Seed busId with the current NIC 0's busId to make discovering a unique hash quicker
        NCCLCHECK(xmlGetAttrStr(pci, "busid", &originalBusId));
        snprintf(busId, sizeof(busId), "%s", originalBusId);
        NCCLCHECK(ncclTopoMakeUniqueBusId(xml, busId, &newBusId, *parent));
        for (int i = 0; i < pci->nAttrs; i++) {
          NCCLCHECK(xmlSetAttr(newBusId, pci->attrs[i].key, pci->attrs[i].value));
        }
        NCCLCHECK(xmlSetAttr(newBusId, "busid", busId));
        *parent = newBusId;
      }
    }
  }

  if (newBusId == NULL) {
    const char* name;
    NCCLCHECK(xmlGetAttr(physNetNode, "name", &name));
    WARN("TOPO/NET : Can't find busId of child 0 %s", name);
    return ncclInternalError;
  }

  return ncclSuccess;
}
//这个函数应该是不完整的，还没完全实现，
ncclResult_t ncclTopoMakeVnic(ncclComm_t comm, struct ncclXml* xml, ncclNetVDeviceProps_t* vProps,
struct ncclXmlNode** physNetNodes, struct ncclXmlNode** netNode, ncclResult_t (*makeVDevice)(int*, ncclNetVDeviceProps_t*)) {
  if (vProps->ndevs > NCCL_NET_MAX_DEVS_PER_NIC) {
    WARN("TOPO/NET : Tried to merge too many NICs. %d > %d", vProps->ndevs, NCCL_NET_MAX_DEVS_PER_NIC);
    return ncclInternalError;
  }

  // Trigger the merge, then get the new device's properties
  int vDevIndex = 0;
  ncclResult_t ret = makeVDevice(&vDevIndex, vProps);//根据指定的属性（ props ），创建一个虚拟网卡（Virtual NIC，简称 VNIC），并返回其设备索引（ d ） 
  if (ret == ncclInvalidUsage) {
    WARN("TOPO/NET : Tried merging multiple devices together and failed. Try setting NCCL_NET_MERGE_LEVEL=LOC");
    NCCLCHECK(ret);
  }

  INFO(NCCL_GRAPH, "TOPO/NET : Made vNic %d", vDevIndex);
  return ncclSuccess;
}

ncclResult_t ncclTopoForceMerge(ncclComm_t comm, struct ncclXml* xml, char* str, int* placedDevs, ncclNetProperties_t* propsList, struct ncclXmlNode** physNetNodes, int nPhysDevs, ncclResult_t (*makeVDevice)(int*, ncclNetVDeviceProps_t*)) {
  INFO(NCCL_ENV|NCCL_NET, "TOPO/NET : Force-fusing NICs using NCCL_NET_FORCE_MERGE=%s", str);
  char* semi_token;
  char* semi = strtok_r(str, ";", &semi_token);
  while (semi) {
    TRACE(NCCL_NET, "Fusing %s", semi);
    struct netIf userIfs[NCCL_NET_MAX_DEVS_PER_NIC];
    int nUserIfs = parseStringList(semi, userIfs, NCCL_NET_MAX_DEVS_PER_NIC);
    if (nUserIfs == 0) {
      INFO(NCCL_NET, "NET/IB : Invalid NCCL_NET_FORCE_MERGE specified %s. Couldn't parse substring %s. Please provide a semicolon-delimited list of comma-delimited NIC groups.",
        str, semi);
      continue;
    }

    ncclNetVDeviceProps_t vProps = {0};
    for (int d = 0; d < nPhysDevs; d++) {
      if (matchIfList(propsList[d].name, propsList[d].port, userIfs, nUserIfs, 1)) {
        vProps.devs[vProps.ndevs++] = d;
      }
    }

    if (vProps.ndevs != nUserIfs) {
      WARN("TOPO/NET : Only matched %d devices, %d requested from %s",
        vProps.ndevs, nUserIfs, semi);
      return ncclInvalidUsage;
    }

    if (vProps.ndevs > NCCL_NET_MAX_DEVS_PER_NIC) {
      WARN("Specified fused NIC %s which has too many devices (%d). Max %d", semi, vProps.ndevs, NCCL_NET_MAX_DEVS_PER_NIC);
      return ncclInvalidUsage;
    }

    struct ncclXmlNode* netNode;
    NCCLCHECK(ncclTopoMakeVnic(comm, xml, &vProps, physNetNodes, &netNode, makeVDevice));

    // Only set that a device is "placed" after successfully making a vNic (it's possible to exit before this)
    for (int i = 0; i < vProps.ndevs; i++) {
      placedDevs[vProps.devs[i]] = 1;
    }

    semi = strtok_r(NULL, ";", &semi_token);;
  }

  return ncclSuccess;
}
//根据物理网卡之间的拓扑距离（路径类型），自动将距离足够近的物理网卡合并为一个虚拟网卡节点，插入到 NCCL 拓扑结构中。
ncclResult_t ncclTopoAutoMerge(ncclComm_t comm, struct ncclXml* xml, int mergeLevel, int* placedDevs, ncclNetProperties_t* propsList, struct ncclXmlNode** physNetNodes, int nPhysDevs, ncclResult_t (*makeVDevice)(int*, ncclNetVDeviceProps_t*)) {
  // Compute the path type between each device
  int* paths = NULL; //为所有物理网卡节点两两之间分配一个路径类型数组 paths ，用于存储它们之间的“距离”或“拓扑路径类型”。
  ncclResult_t res = ncclSuccess;
  ncclCalloc(&paths, nPhysDevs*nPhysDevs);
  TRACE(NCCL_GRAPH, "Allocated %d paths", nPhysDevs*nPhysDevs);
  for (int i = 0; i < nPhysDevs; i++) {
    for (int j = 0; j < nPhysDevs; j++) {
      struct ncclXmlNode* nodes[2];
      nodes[0] = physNetNodes[i];
      nodes[1] = physNetNodes[j];
      struct ncclXmlNode* parent;
      NCCLCHECKGOTO(ncclTopoGetPath(nodes, 2, &paths[i*nPhysDevs + j], &parent), res, out);//计算每对物理网卡节点之间的路径类型
      //这样后续可以根据路径类型决定哪些物理网卡可以合并为一个虚拟网卡
    }
  }
 
  //合并物理网卡为虚拟网卡

  // Place all remaining physical devices into a virtual device given the mergeLevel criteria
  for (int i = 0; i < nPhysDevs; i++) {
    // Select the first unplaced device "i" as the root
    //外层循环遍历所有物理网卡，找到第一个还未被合并（ placedDevs[i] == 0 ）的设备 i，作为新虚拟网卡的“根”
    if (placedDevs[i] == 0) {
      // Init a new vDevice
      ncclNetVDeviceProps_t vProps; //初始化一个新的虚拟网卡属性结构体 vProps ，并把 i 加入其中，标记为已合并。
      vProps = {0};
      vProps.devs[vProps.ndevs++] = i;//设备ID为i
      placedDevs[i] = 1;//标记为已合并。
      TRACE(NCCL_GRAPH, "Placed dev %d", i);

      // Select each unplaced device "j" which is at most "mergeLevel" distance from "i", but not equal to "i"
      // (Don't merge the same device with itself)
      //内层循环遍历所有未被合并的设备 j，如果 j 到 i 的路径类型小于等于 mergeLevel （即拓扑距离足够近），则也加入到当前虚拟网卡中，并标记为已合并。
      for (int j = 0; j < nPhysDevs; j++) {
        if (paths[i*nPhysDevs + j] <= mergeLevel &&
        placedDevs[j] == 0 && j != i) {
          vProps.devs[vProps.ndevs++] = j;
          placedDevs[j] = 1;
          TRACE(NCCL_GRAPH, "Placed dev %d path=%d", j, paths[i*nPhysDevs + j] );
        }
        if (vProps.ndevs == NCCL_NET_MAX_DEVS_PER_NIC) break;//每个虚拟网卡最多合并 NCCL_NET_MAX_DEVS_PER_NIC 个物理网卡，超过则报错。
      }

      if (vProps.ndevs > NCCL_NET_MAX_DEVS_PER_NIC) {
        WARN("TOPO/NET : Tried to merge too many NICs. %d > %d", vProps.ndevs, NCCL_NET_MAX_DEVS_PER_NIC);
        return ncclInternalError;
      }
      //合并完成后，调用 ncclTopoMakeVnic 创建虚拟网卡节点，并插入到拓扑结构中。
      struct ncclXmlNode* netNode;
      NCCLCHECKGOTO(ncclTopoMakeVnic(comm, xml, &vProps, physNetNodes, &netNode, makeVDevice), res, out);
    }
  }

out:
  free(paths);
  return res;
}

struct kvDict nicPathKvList[] = {
  { "LOC",  PATH_LOC },
  { "PORT", PATH_PORT },
  { "PIX",  PATH_PIX },
  { "PXB",  PATH_PXB },
  { "PXN",  PATH_PXN },
  { "PHB",  PATH_PHB },
  { "SYS",  PATH_SYS },
  { NULL, 0 }
};
//为虚拟网卡（vNIC）在 NCCL 拓扑 XML 中找到合适的父节点（parent），以便正确插入到系统拓扑结构中。
// 如果需要，还会为虚拟网卡构造一个虚拟的 PCI 父节点，保证拓扑结构的合理性和一致性。

ncclResult_t ncclTopoGetVNicParent(struct ncclXml* xml, ncclResult_t (*getProperties)(int, ncclNetProperties_t*), ncclNetVDeviceProps_t* vProps, ncclXmlNode** parent) {
  ncclNetProperties_t props[NCCL_NET_MAX_DEVS_PER_NIC];
  ncclXmlNode* physNetNodes[NCCL_NET_MAX_DEVS_PER_NIC];
  //获取每个物理网卡的属性，并在 XML 中查找对应的 net 节点，保存到 physNetNodes 数组。
  

  for (int i = 0; i < vProps->ndevs; i++) {
    NCCLCHECK(getProperties(vProps->devs[i], props + i));
    struct ncclXmlNode* physNetNode;
    NCCLCHECK(xmlFindTagKv(xml, "net", &physNetNode, "name", props[i].name));
    physNetNodes[i] = physNetNode;
    TRACE(NCCL_GRAPH, "Re-found physical ncclNet node %d %s", i,  props[i].name);
  }

  int path = PATH_LOC;
  //计算这些物理网卡节点的共同父节点（parent），以及它们之间的路径类型（path）。
  NCCLCHECK(ncclTopoGetPath(physNetNodes, vProps->ndevs, &path, parent));
  if (path == PATH_LOC) {//这表示所有物理网卡的共同父节点就是它们自己（即只有一个节点），此时不需要再为虚拟网卡指定父节点
    *parent = NULL;
  } else if (parent && strcmp((*parent)->name, "pci") == 0) {
    //如果找到的 parent 节点是 PCI 类型，则需要调用 ncclTopoMakePciParent ，
    // 为新的虚拟网卡生成一个虚拟的 PCI 父节点（因为多个物理网卡的共同父节点是 PCI，需要人为构造一个 busId 以便插入）
    // If the common parent is PCI, we must reparent the new NIC under a made up busId
    NCCLCHECK(ncclTopoMakePciParent(xml, parent, physNetNodes[0]));
  }
  TRACE(NCCL_GRAPH, "Selected parent %s with path %d", (*parent)->name, path);
  return ncclSuccess;
}
//根据物理网卡的实际情况和环境变量配置，将物理网卡合并为虚拟网卡，并在拓扑结构中正确插入这些虚拟网卡节点。
ncclResult_t ncclTopoMakeVNics(ncclComm_t comm, struct ncclXml* xml, ncclResult_t (*makeVDevice)(int*, ncclNetVDeviceProps_t*), ncclResult_t (*getProperties)(int, ncclNetProperties_t*), int physicalDevs) {
  int* placedDevs = NULL;
  struct ncclXmlNode** physNetNodes = NULL;
  if (physicalDevs == 0) return ncclSuccess;

  ncclCalloc(&physNetNodes, physicalDevs);
  ncclResult_t res = ncclSuccess;

  ncclNetProperties_t* props = NULL;
  ncclCalloc(&props, physicalDevs);
  //遍历所有物理网卡，获取属性并定位 XML 节点。
  for (int i = 0; i < physicalDevs; i++) {
    NCCLCHECKGOTO(getProperties(i, props + i), res, out);
    struct ncclXmlNode* physNetNode;
    NCCLCHECKGOTO(xmlFindTagKv(xml, "net", &physNetNode, "name", props[i].name), res, out);
    physNetNodes[i] = physNetNode;
    TRACE(NCCL_GRAPH, "Found physical ncclNet node %d %s", i,  props[i].name);
  }

  // By default, don't merge any devices. 决定虚拟网卡合并的粒度（如端口级、设备级等），可通过环境变量覆盖
  int mergeLevel;
  mergeLevel = PATH_PORT;
  char* mergeLevelEnv;
  mergeLevelEnv = getenv("NCCL_NET_MERGE_LEVEL");
  if (mergeLevelEnv) kvConvertToInt(mergeLevelEnv, &mergeLevel, nicPathKvList);
  char* forceMerge;
  forceMerge = getenv("NCCL_NET_FORCE_MERGE");//如果设置了该环境变量，则强制按照指定方式合并
  NCCLCHECK(ncclCalloc(&placedDevs, physicalDevs));
  memset(placedDevs, 0, sizeof(int)*physicalDevs);//用于标记哪些物理网卡已经被合并进虚拟网卡，防止重复处理

  if (forceMerge) {
    NCCLCHECKGOTO(ncclTopoForceMerge(comm, xml, forceMerge, placedDevs, props, physNetNodes, physicalDevs, makeVDevice), res, out);
  }
  NCCLCHECKGOTO(ncclTopoAutoMerge(comm, xml, mergeLevel, placedDevs, props, physNetNodes, physicalDevs, makeVDevice), res, out);

out:
  free(physNetNodes);
  free(props);
  if (placedDevs) free(placedDevs);
  return res;
}
/*
根据网卡属性，填充或更新 NCCL 拓扑 XML 中的 net 节点，并设置其 keep、coll 等属性，决定该节点是否参与后续通信。
- keep=1 ：表示该网卡节点需要被保留，后续拓扑构建、调度等都会考虑它。
- keep=0 ：表示该网卡节点可以被忽略或删除，不参与后续的通信拓扑。
*/
static ncclResult_t ncclTopoPopulateNics(ncclComm_t comm, ncclXml* xml, int startIndex, int endIndex, ncclResult_t (*getProperties)(int, ncclNetProperties_t*), const char* netName, int coll, int keep, int virtualNics) {
  for (int n = startIndex; n < endIndex; n++) {
    ncclNetProperties_t props;
    NCCLCHECK(getProperties(n, &props));//第 n 个网卡的属性，填充到 props 结构体。
    struct ncclXmlNode* netNode = NULL;
    struct ncclXmlNode* parent = NULL;
    if (virtualNics) {
      //如果是虚拟网卡（ virtualNics 为真），尝试查找 XML 中是否已有对应节点，如果没有则查找其父节点。
      struct ncclXmlNode* net = NULL;
      NCCLCHECK(xmlFindTagKv(xml, "net", &net, "name", props.name));
      // In the event of multithreaded use case, we need to re-discover the shared parent of the given devices for this vNIC
      // Only run this if the net doesn't exist locally - this may alter the XML state 在多线程场景下，需要重新发现该 vNIC 设备的共享父节点
      // 这是因为在多线程环境下，多个线程可能会同时操作 XML 拓扑结构，导致某些 net 节点还未被创建或已被修改。此时需要重新查找或创建合适的父节点，确保拓扑结构的正确性和一致性。
      // 虚拟网卡（vNIC）本身并不是独立存在于硬件中的设备，它们通常是基于某个物理网卡（NIC）或PCI设备虚拟出来的。为了让整个系统的拓扑结构保持一致性和可用性，
      // 必须把这些vNIC正确地“挂载”到它们所属的物理父节点（如物理NIC、PCIe桥、CPU等）下。这样才能反映出它们和物理设备之间的真实关系。
      if (net == NULL) NCCLCHECK(ncclTopoGetVNicParent(xml, getProperties, &props.vProps, &parent));
    }
    //用于在 XML 拓扑中找到或创建对应的 net 节点，并且挂载到合适的父节点下（大概是CPU->PCI->NIC->Net）
    NCCLCHECK(ncclTopoFillNet(xml, props.pciPath, props.name, &netNode, parent));

    const char* colAttr;
    NCCLCHECK(xmlGetAttr(netNode, "coll", &colAttr));//coll 属性获取

    // If coll == 0 but the netNode is tagged as coll, don't update the keep value.
    // 这句的意思是：如果 netNode 没有 coll 属性，或者 coll 不是0，或者 collAttr不是"1"，就设置 keep 属性。这样可以避免在某些特殊情况下覆盖 keep 的值。
    // 这句代码的作用就是 保护 collective NIC 的 keep 属性不被普通 NIC 覆盖 ，确保拓扑中 collective NIC 的特殊属性不会被误改。
    //keep 属性一般用于标记该网卡节点在后续拓扑优化、裁剪、筛选等流程中是否应该被保留。如果 collective NIC 的 keep 属性被普通 NIC 的逻辑覆盖，可能导致本应保留的高性能网卡被错误地移除或忽略。
    // 因为函数参数的keep是应用于所有网卡的，但是对于高性能网卡，我们希望保留他的keep属性。
    if (colAttr == NULL || coll != 0 || strcmp(colAttr,"1") != 0) NCCLCHECK(xmlSetAttrInt(netNode, "keep", keep));
    NCCLCHECK(xmlSetAttrInt(netNode, "dev", n));
    NCCLCHECK(xmlInitAttrInt(netNode, "latency", props.latency));
    NCCLCHECK(xmlInitAttrInt(netNode, "speed", props.speed));
    NCCLCHECK(xmlInitAttrInt(netNode, "port", props.port));
    NCCLCHECK(xmlInitAttrUint64(netNode, "guid", props.guid));
    NCCLCHECK(xmlInitAttrInt(netNode, "maxconn", props.maxComms));
    // 判断该网卡是否支持 GPU Direct RDMA（GDR
    bool gdrSupport = (props.ptrSupport & NCCL_PTR_CUDA) || (comm->dmaBufSupport && (props.ptrSupport & NCCL_PTR_DMABUF));
    INFO(NCCL_NET,"NET/%s : GPU Direct RDMA %s for HCA %d '%s'", netName, gdrSupport ? "Enabled" : "Disabled", n, props.name);
    NCCLCHECK(xmlInitAttrInt(netNode, "gdr", gdrSupport));
    // Only set coll if it's not 0
    if (coll) NCCLCHECK(xmlInitAttrInt(netNode, "coll", coll));

    const char* keepAttr;
    NCCLCHECK(xmlGetAttr(netNode, "coll", &colAttr));
    NCCLCHECK(xmlGetAttr(netNode, "keep", &keepAttr));
    INFO(NCCL_GRAPH, "ncclTopoPopulateNics : Filled %s in topo with pciPath=%s keep=%s coll=%s",
      props.name, props.pciPath, keepAttr, colAttr);
  }

  return ncclSuccess;
}

struct ncclTopoNetState {
  int nVirtualNics; // 虚拟网卡数量（如 SR-IOV 场景下的虚拟 NIC 数）
  int nPhysicalNics;// 物理网卡数量
  const char* name; // 网络插件的名称（如 "IB", "RoCE", "TCP" 等）
};

// Calls to network plugin APIs should be protected. This function should be called inside a per-process lock.
//此函数应在进程级别的锁保护下调用，防止多线程并发访问网络插件 API 导致竞态。
//用于处理 NCCL 拓扑中的网络插件（如 IB、RoCE、TCP 等），负责枚举物理/虚拟网卡，并将其信息填充到 NCCL 拓扑结构中。
static ncclResult_t ncclTopoProcessNet(ncclComm_t comm, ncclXml* xml, int coll, const char* dumpXmlFile, ncclTopoNetState* state, ncclResult_t (*getProperties)(int, ncclNetProperties_t*), ncclResult_t (*makeVDevice)(int*, ncclNetVDeviceProps_t*), ncclResult_t (*devices)(int*), const char* netName) {
  //如果需要导出 XML 文件，或者没有虚拟网卡创建函数，则只使用物理网卡。
  //导出 XML 文件的目的是记录和反映当前系统的真实硬件拓扑结构，包括物理存在的 CPU、GPU、NIC（网卡）等设备，以及它们之间的连接关系。
  // 虚拟网卡（如 SR-IOV、软件模拟的 vNIC）是软件层面动态生成的，属于运行时优化或资源复用的产物，并不反映真实的硬件结构。
  int usePhysicalDevices = (dumpXmlFile || makeVDevice == NULL);
  if (state->nPhysicalNics == -1) NCCLCHECK(devices(&state->nPhysicalNics));//获取物理网卡数量
  // Enumerate physical devices 遍历所有物理网卡，将其属性填充到 NCCL 拓扑结构中。注意这里设置了vnic为0，表示先不处理（因为要先获取所有物理网卡的信息，添加到xml中后才会后续填补vnic的细节。）
  NCCLCHECK(ncclTopoPopulateNics(comm, xml, 0, state->nPhysicalNics, getProperties, netName, coll, 1, 0));
  if (!usePhysicalDevices) {//如果允许使用虚拟网卡
    if (state->nVirtualNics == -1) {
      //创建虚拟网卡
      NCCLCHECK(ncclTopoMakeVNics(comm, xml, makeVDevice, getProperties, state->nPhysicalNics));
      int nDevs;
      NCCLCHECK(devices(&nDevs));
      state->nVirtualNics = nDevs - state->nPhysicalNics;//虚拟网卡数 = 总数 - 物理网卡数。
    }
    // Remove keep=1 for physical collnets
    if (state->nVirtualNics > 0) {
      //先移除物理 collnet 的 keep 标记（ keep=1 ），即物理网卡不再作为 collnet 设备。
      NCCLCHECK(ncclTopoPopulateNics(comm, xml, 0, state->nPhysicalNics, getProperties, netName, coll, 0, 0));
      // Populate new devices 然后枚举并填充所有虚拟网卡到拓扑结构中。
      NCCLCHECK(ncclTopoPopulateNics(comm, xml, state->nPhysicalNics, state->nPhysicalNics+state->nVirtualNics, getProperties, netName, coll, 1, 1));
    }
  }

  return ncclSuccess;
}

static pthread_mutex_t netLock = PTHREAD_MUTEX_INITIALIZER;
//共享的网络状态数组，用于存储每个网络插件的状态信息。
ncclTopoNetState netStates[NCCL_NET_MAX_PLUGINS] = {};
ncclTopoNetState collNetStates[NCCL_NET_MAX_PLUGINS] = {};
//为指定名字的网络插件（如 IB、RoCE、TCP 等）在全局状态数组中分配或查找一个共享状态槽，并返回其指针。
ncclResult_t ncclTopoGetSharedState(ncclTopoNetState** state, const char* name, ncclTopoNetState* states) {
  INFO(NCCL_GRAPH, "Retrieving state for %s", name);
  for (int i = 0; i < NCCL_NET_MAX_PLUGINS; i++) {
    // Empty slot 如果当前槽位还没有被占用（即还没有分配给任何插件），就初始化它：
    if (states[i].name == NULL) {
      states[i].nVirtualNics = -1;
      states[i].nPhysicalNics = -1;
      states[i].name = strdup(name);
      *state = states + i;
      INFO(NCCL_GRAPH, "Initialized state %d for %s", i, name);
      return ncclSuccess;
    // Found my slot
    } else if (strcmp(states[i].name, name) == 0) {
      *state = states + i;
      return ncclSuccess;
    }
  }
  WARN("NET/TOPO : Couldn't find net with name %s", name);
  return ncclInternalError;
}
//根据 NCCL 通信器 comm ，构建当前节点的完整拓扑系统结构 system ，并支持 XML 拓扑导入、自动检测、融合、导出等功能。
ncclResult_t ncclTopoGetSystem(struct ncclComm* comm, struct ncclTopoSystem** system, const char* dumpXmlFile) {
  ncclResult_t ret = ncclSuccess;
  struct ncclXml* xml;
  char* mem = NULL;
  int* localRanks = NULL;
  struct ncclXml* rankXml;
  int localRank = -1, nLocalRanks = 0;
  int netLockHeld = 0;
  NCCLCHECK(xmlAlloc(&xml, NCCL_TOPO_XML_MAX_NODES));
  const char* xmlTopoFile = ncclGetEnv("NCCL_TOPO_FILE");//优先从环境变量 NCCL_TOPO_FILE 指定的文件加载 XML 拓扑。
  if (xmlTopoFile) {
    INFO(NCCL_ENV, "NCCL_TOPO_FILE set by environment to %s", xmlTopoFile);
    NCCLCHECKGOTO(ncclTopoGetXmlFromFile(xmlTopoFile, xml, 1), ret, fail);
  } else {
    // Try default XML topology location 尝试加载默认路径
    NCCLCHECKGOTO(ncclTopoGetXmlFromFile("/var/run/nvidia-topologyd/virtualTopology.xml", xml, 0), ret, fail);
  }
  if (xml->maxIndex == 0) {
    // Create top tag 如果 XML 还没有任何节点，则新建一个名为 system 的根节点，并设置版本号。
    struct ncclXmlNode* top;
    NCCLCHECKGOTO(xmlAddNode(xml, NULL, "system", &top), ret, fail);
    NCCLCHECKGOTO(xmlSetAttrInt(top, "version", NCCL_TOPO_XML_VERSION), ret, fail);
  }
  //刷新 PCIe 交换机的 P2P 拓扑信息（如有需要）。
  NCCLCHECKGOTO(ncclTopoRefreshBcmP2pLinks(), ret, fail);

  // Detect only the GPU managed by this process.  We'll get any others through XML fusion.
  //检测本进程管理的 GPU 并标记。其他的GPU信息可以通过XML融合获得。
  char busId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];//busId在分配comm内存的时候就已经获得了
  //注意这里的busId不是GPU的busId，而是这个槽的信息。与具体挂载的设备无关
  //PCI 节点（busId）是插槽/端口的抽象，具体设备信息（如 GPU）是其子节点或附加属性
  NCCLCHECKGOTO(int64ToBusId(comm->peerInfo[comm->rank].busId, busId), ret, fail);
  struct ncclXmlNode* node;
  NCCLCHECKGOTO(ncclTopoFillGpu(xml, busId, &node), ret, fail);//将本rank的gpu信息填入到xml中,即在后续的 XML 拓扑精简（如 ncclTopoTrimXml ）过程中，这个节点及其相关分支不会被删除。
  if (node) {
    NCCLCHECKGOTO(xmlSetAttrInt(node, "keep", 1), ret, fail);//标记当前节点（本进程管理的 GPU 节点）为“需要保留”
    NCCLCHECKGOTO(xmlSetAttrInt(node, "rank", comm->rank), ret, fail);
    NCCLCHECKGOTO(xmlInitAttrInt(node, "gdr", comm->peerInfo[comm->rank].gdrSupport), ret, fail);
  }

  // Auto-detect NICs if needed. net/collnet share the same xml/graph nodes,
  // so we start with collnet so that it has precedence.
  //加锁，保证后续网络拓扑的检测和导入过程是线程安全的，防止多线程同时修改网络相关结构。
  //只有网络插件相关的拓扑导入需要全局加锁，是因为它们涉及全局共享状态和资源，其他部分则不需要。
  /*
  NCCL 支持多种网络插件（如 IB、RoCE、CollNet 等），这些插件的初始化、状态（如 netStates 、 collNetStates ）和拓扑信息是全局共享的。
  如果多个线程同时导入或修改这些状态，可能会导致竞态条件、内存泄漏或数据不一致。
  NCCL 的 PCI/CPU/GPU 拓扑结构一般在每个进程内部独立构建，不涉及全局共享状态，因此不需要加锁。

  比如在一台机器上有多块网卡（NIC），NCCL 需要确保所有进程看到的网络设备列表是一致的，不能每个线程/进程各自枚举一遍，
  否则会导致设备编号、属性等不一致，影响通信拓扑的正确性。
  */
  pthread_mutex_lock(&netLock);
  //标记当前线程已经持有网络相关的互斥锁。
  netLockHeld = 1;
  INFO(NCCL_GRAPH, "TOPO/NET : Importing network plugins to topology");
  ncclTopoNetState* state;
  state = NULL;
  if (collNetSupport(comm)) {//支持CollNet
    //获取 CollNet 插件的共享状态。例如IB、RoCE等。
    NCCLCHECKGOTO(ncclTopoGetSharedState(&state, comm->ncclCollNet->name, collNetStates), ret, fail);
    // 处理 CollNet 网络，将其信息导入到 NCCL 拓扑结构中。参数 1 表示是 CollNet。
    NCCLCHECKGOTO(ncclTopoProcessNet(comm, xml, 1, dumpXmlFile, state,
      comm->ncclCollNet->getProperties, comm->ncclCollNet->makeVDevice, comm->ncclCollNet->devices, comm->ncclCollNet->name), ret, fail);
  }
  //获取并处理网络相关的共享状态（如多网卡、虚拟网卡等）。
  NCCLCHECKGOTO(ncclTopoGetSharedState(&state, comm->ncclNet->name, netStates), ret, fail);
  //负责根据网络属性、设备信息等，处理 XML 拓扑结构，生成合适的网络节点
  NCCLCHECKGOTO(ncclTopoProcessNet(comm, xml, 0, dumpXmlFile, state,
    comm->ncclNet->getProperties, comm->ncclNet->makeVDevice, comm->ncclNet->devices, comm->ncclNet->name), ret, fail);
  pthread_mutex_unlock(&netLock);
  netLockHeld = 0;

  // Remove XML branches which don't have a node with keep="1" (typically when importing a topology)
  // 移除 XML 中没有 keep="1" 标记的分支（通常是导入拓扑时的冗余节点），保证后续处理的 XML 拓扑干净、有效。
  NCCLCHECKGOTO(ncclTopoTrimXml(xml), ret, fail);

  // XML topo fusion. 

  /*
关于融合：
NCCL 的主要优化目标是节点内的 GPU 通信（如 NVLink、PCIe），而节点间通信受限于网络瓶颈，优化空间有限。只融合节点内 XML 能满足绝大多数高性能通信需求。
集群中每个节点（主机）之间通过网络（如 InfiniBand、Ethernet）连接，而节点内的 GPU/CPU/PCIe/NVLink 拓扑才是真正需要详细建模和优化的部分。
节点间的网络连接通常被抽象为“NET”节点，带宽、延迟等参数远低于节点内互联，且网络结构复杂多变，难以用统一的 XML 拓扑描述。
如果把整个集群的 XML 融合在一起，随着节点数增加，XML 文件会变得极其庞大，解析、同步和维护都非常低效。而节点内 XML 只需描述本机的硬件结构，体积小、处理快，易于并行化。

  - 判断当前是否为 MNNVL（多节点 NVLink ）模式。
    - 如果是，直接用 clique 信息（clique 是一组互联的 GPU）。
    - 如果不是，遍历所有 rank，找出和本地 rank 在同一主机（hostHash 相同）的 rank，构建 localRanks 数组。

  这样后续可以针对 clique 内的所有进程进行拓扑融合、通信优化等操作。
  */
  if (comm->MNNVL) {
    // MNNVL clique support
    nLocalRanks = comm->clique.size;//这表示本地“clique”（团，完全互联子集）的进程（rank）数量
    localRank = comm->cliqueRank;//这是当前进程（rank）在 clique 内的编号（索引）
    localRanks = comm->clique.ranks;//这是一个数组，包含了所有属于同一个 clique 的 rank (全局的)编号。
  } else {
    // Intra-node fusion.  Much of the comm is not initialized yet at this point so we need to do our own calculations.
    //在节点内做拓扑融合时，由于 NCCL 的通信结构体还没准备好，所以需要手动计算哪些 rank 属于本地节点，不能直接用 comm 里的现成数据。
    //虽然 bootstrap（如 bootstrapAllGather、bootstrapIntraNodeAllGather 等）已经完成了进程间的基本通信能力建立，
    // 但 NCCL 的 comm 结构体（尤其是其中的 localRanks、localRank、clique 等成员）并不是在 bootstrap 完成后立刻全部填充好的。
    //bootstrap 只负责进程间通信能力的建立,comm 结构体的部分成员需要依赖后续的拓扑融合.只有等到所有本地 rank 的 XML 拓扑都融合完毕，
    // 才能最终确定哪些 rank 属于本地节点、它们的编号等信息，然后再填充到 comm 里。
    // 如果不是，遍历所有 rank，找出和本地 rank 在同一主机（hostHash 相同）的 rank，构建 localRanks 数组。
    NCCLCHECKGOTO(ncclCalloc(&localRanks, comm->nRanks), ret, fail);
    for (int i = 0; i < comm->nRanks; i++) {
      if (comm->peerInfo[i].hostHash == comm->peerInfo[comm->rank].hostHash) {
        if (i == comm->rank)
          localRank = nLocalRanks;
        localRanks[nLocalRanks++] = i;
      }
    }
  }
  // 为每个本地 rank 分配一份 XML 拓扑内存。
  NCCLCHECKGOTO(ncclCalloc(&mem, nLocalRanks * xmlMemSize(NCCL_TOPO_XML_MAX_NODES)), ret, fail);
  // 将本地 XML 拓扑复制到对应的内存区域，并做一次转换（如指针修正等）。
  rankXml = (struct ncclXml*)(mem+xmlMemSize(NCCL_TOPO_XML_MAX_NODES)*localRank);
  memcpy(rankXml, xml, xmlMemSize(NCCL_TOPO_XML_MAX_NODES));
  //对 NCCL 内部用于描述硬件拓扑的 XML 结构体中的指针进行“指针序列化/反序列化”操作。把每个节点的 parent 和 subs （子节点）指针，转换为“偏移量”或“恢复为指针
  NCCLCHECKGOTO(ncclTopoConvertXml(rankXml, (uintptr_t)xml->nodes, 1), ret, fail);
  // nLocalRanks can't actually be 0, or we wouldn't be running at all...
  // coverity[divide_by_zero]
  // 通过 bootstrapIntraNodeAllGather，把所有本地 rank 的 XML 拓扑信息同步到每个进程（rank）本地，实现“全员可见”。
  NCCLCHECKGOTO(bootstrapIntraNodeAllGather(comm->bootstrap, localRanks, localRank, nLocalRanks, mem, xmlMemSize(NCCL_TOPO_XML_MAX_NODES)), ret, fail);
  if (comm->MNNVL) {
    // Ensure that we have enough room when fusing topos from multiple nodes.
    // 如果是多节点 NVLink 融合，需要为融合后的大 XML 分配更大空间。
    free(xml);
    xml = NULL;
    NCCLCHECKGOTO(xmlAlloc(&xml, nLocalRanks*NCCL_TOPO_XML_MAX_NODES), ret, fail);
  } else {
    // In the intra-node case there's no need to enlarge the topo xml.
    //单节点场景则不需要扩容。只是重置一下节点数量（复用之前的xml结构体）。
    xml->maxIndex = 0;
  }
  for (int i = 0; i < nLocalRanks; i++) {
    //遍历所有本地 rank 的 XML 拓扑，依次融合到主 XML 结构中。
    struct ncclXml* peerXml = (struct ncclXml*)(mem+xmlMemSize(NCCL_TOPO_XML_MAX_NODES)*i);
    NCCLCHECKGOTO(ncclTopoConvertXml(peerXml, (uintptr_t)peerXml->nodes, 0), ret, fail);//反序列化。
    NCCLCHECKGOTO(ncclTopoFuseXml(xml, peerXml), ret, fail);//把 peerXml 的节点合并进 xml。
  }
  // 如果设置了环境变量要求导出 XML 拓扑，并且当前 rank 是指定的导出 rank，则将融合后的 XML 拓扑写入文件。
  if (dumpXmlFile && comm->rank == ncclParamTopoDumpFileRank()) {
    INFO(NCCL_ENV, "NCCL_TOPO_DUMP_FILE set by environment to %s", dumpXmlFile);
    NCCLCHECKGOTO(ncclTopoDumpXmlToFile(dumpXmlFile, xml), ret, fail);
  }

  // Only update our topo tracking structure if we aren't dumping (separate steps)
  // 如果不是仅仅导出 XML 文件，则根据融合后的 XML 拓扑，生成最终的 NCCL 拓扑系统结构体（system）。
  if (dumpXmlFile == NULL) NCCLCHECKGOTO(ncclTopoGetSystemFromXml(xml, system, comm->peerInfo[comm->rank].hostHash), ret, fail);

exit:
  if (!comm->MNNVL && localRanks) free(localRanks);
  if (mem) free(mem);
  free(xml);
  return ret;
fail:
  if (netLockHeld) pthread_mutex_unlock(&netLock);
  goto exit;
}

static ncclResult_t ncclTopoGetLocal(struct ncclTopoSystem* system, int type, int index, int resultType,
                                     int locals[NCCL_TOPO_MAX_NODES], int* localCount, int* pathType) {
  int minType = PATH_DIS;
  float maxBw = 0;
  int count = 0;
  struct ncclTopoLinkList* paths = system->nodes[type].nodes[index].paths[resultType];
  if (paths == NULL) { *localCount = 0; return ncclSuccess; }
  for (int i=0; i<system->nodes[resultType].count; i++) {
    if (paths[i].bw > maxBw || (paths[i].bw == maxBw && paths[i].type < minType)) {
      maxBw = paths[i].bw;
      minType = paths[i].type;
      if (pathType) *pathType = minType;
      count = 0;
    }
    if (paths[i].bw == maxBw && paths[i].type == minType) {
      if (count == NCCL_TOPO_MAX_NODES) {
        WARN("Error : ran out of room to store found nodes in ncclTopoGetLocal."
             " Filled %d of type %d, starting from index %d of type %d.",
             NCCL_TOPO_MAX_NODES, resultType, index, type);
        return ncclInternalError;
      }
      locals[count++] = i;
    }
  }
  *localCount = count;
  return ncclSuccess;
}

ncclResult_t getLocalNetCountByBw(struct ncclTopoSystem* system, int gpu, int *count) {
  int localNetCount = 0, netCountByBw = 0;
  int localNets[NCCL_TOPO_MAX_NODES];
  float totalNetBw = 0, gpuBw = 0;

  for (int l=0; l<system->nodes[GPU].nodes[gpu].nlinks; l++) {
    //assuming BW to CPU reflects the GPU bandwidth via P2P or C2C
    //caveat, this could be wrong if there is a PCIe switch,
    //and a narrower link to the CPU
    if (system->nodes[GPU].nodes[gpu].links[l].remNode->type == CPU) {
       gpuBw = system->nodes[GPU].nodes[gpu].links[l].bw;
    }
  }

  NCCLCHECK(ncclTopoGetLocal(system, GPU, gpu, NET, localNets, &localNetCount, NULL));
  for (int l=0; (l < localNetCount) && (totalNetBw < gpuBw); l++, netCountByBw++) {
     totalNetBw += system->nodes[GPU].nodes[gpu].paths[NET][localNets[l]].bw;
  }
  *count = netCountByBw;

  return ncclSuccess;
}

ncclResult_t ncclTopoGetLocalNet(struct ncclTopoSystem* system, int rank, int channelId, int64_t* id, int* dev) {
  int gpu;
  NCCLCHECK(ncclTopoRankToIndex(system, rank, &gpu));

  int localNets[NCCL_TOPO_MAX_NODES];
  int localNetCount;
  NCCLCHECK(ncclTopoGetLocal(system, GPU, gpu, NET, localNets, &localNetCount, NULL));
  if (localNetCount==0) {
    WARN("Could not find any local path from gpu %d to net.", gpu);
    return ncclInternalError;
  }

  int localGpus[NCCL_TOPO_MAX_NODES];
  int localGpuCount;
  NCCLCHECK(ncclTopoGetLocal(system, NET, localNets[0], GPU, localGpus, &localGpuCount, NULL));

  int net = system->nodes[GPU].nodes[gpu].gpu.dev;
  if (isPow2(localNetCount)) net = mirrorBits(net, localNetCount);
  net += channelId%(DIVUP(localNetCount,localGpuCount));
  if (id) *id = system->nodes[NET].nodes[localNets[net%localNetCount]].id;
  if (dev) *dev = system->nodes[NET].nodes[localNets[net%localNetCount]].net.dev;
  return ncclSuccess;
}

ncclResult_t ncclTopoGetLocalGpu(struct ncclTopoSystem* system, int64_t netId, int* gpuIndex) {
  ncclResult_t ret = ncclSuccess;
  int netIndex;
  NCCLCHECK(ncclTopoIdToIndex(system, NET, netId, &netIndex));

  int localGpus[NCCL_TOPO_MAX_NODES];
  int localGpuCount;
  NCCLCHECK(ncclTopoGetLocal(system, NET, netIndex, GPU, localGpus, &localGpuCount, NULL));

  int foundGpu = -1;
  for (int c=0; c<MAXCHANNELS; c++) {
    for (int lg=0; lg<localGpuCount; lg++) {
      int g = localGpus[lg];
      struct ncclTopoNode* gpu = system->nodes[GPU].nodes+g;
      int64_t id;
      NCCLCHECK(ncclTopoGetLocalNet(system, gpu->gpu.rank, c, &id, NULL));
      if (netId == id) {
        foundGpu = g;
        goto exit;
      }
    }
  }
exit:
  *gpuIndex = foundGpu;
  return ret;
}

/****************************/
/* External query functions */
/****************************/

ncclResult_t ncclTopoCpuType(struct ncclTopoSystem* system, int* arch, int* vendor, int* model) {
  *arch = system->nodes[CPU].nodes[0].cpu.arch;
  *vendor = system->nodes[CPU].nodes[0].cpu.vendor;
  *model = system->nodes[CPU].nodes[0].cpu.model;
  return ncclSuccess;
}

NCCL_PARAM(IgnoreCpuAffinity, "IGNORE_CPU_AFFINITY", 0);

ncclResult_t ncclTopoGetCpuAffinity(struct ncclTopoSystem* system, int rank, cpu_set_t* affinity) {
  struct ncclTopoNode* cpu = NULL, *gpu = NULL;
  int gpuIndex, cpuIndex;
  NCCLCHECK(ncclTopoRankToIndex(system, rank, &gpuIndex));
  NCCLCHECK(ncclGetLocalCpu(system, gpuIndex, &cpuIndex));
  gpu = system->nodes[GPU].nodes+gpuIndex;
  cpu = system->nodes[CPU].nodes+cpuIndex;

  // Query the CPU affinity set we were provided
  cpu_set_t mask;
  SYSCHECK(sched_getaffinity(0, sizeof(cpu_set_t), &mask), "sched_getaffinity");

#ifdef ENABLE_TRACE
  {
    char affinityStr[sizeof(cpu_set_t)*2];
    NCCLCHECK(ncclCpusetToStr(&mask, affinityStr));
    TRACE(NCCL_INIT, "Current affinity for GPU %d is %s", gpu->gpu.dev, affinityStr);
  }
#endif

  // Get the affinity of the CPU close to our GPU.
  cpu_set_t cpuMask = cpu->cpu.affinity;

#ifdef ENABLE_TRACE
  {
    char affinityStr[sizeof(cpu_set_t)*2];
    NCCLCHECK(ncclCpusetToStr(&cpuMask, affinityStr));
    TRACE(NCCL_INIT, "CPU GPU affinity for GPU %d is %s", gpu->gpu.dev, affinityStr);
  }
#endif

  cpu_set_t finalMask;
  if (ncclParamIgnoreCpuAffinity())
    // Ignore the CPU affinity set and use the GPU one instead
    finalMask = cpuMask;
  else
    // Use a subset of the GPU affinity set
    CPU_AND(&finalMask, &mask, &cpuMask);

  memcpy(affinity, &finalMask, sizeof(cpu_set_t));

  // If there is a non empty set, use it to set affinity
  if (CPU_COUNT(&finalMask)) {
    char affinityStr[sizeof(cpu_set_t)*2];
    NCCLCHECK(ncclCpusetToStr(&finalMask, affinityStr));
    INFO(NCCL_INIT, "Setting affinity for GPU %d to %s", gpu->gpu.dev, affinityStr);
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoGetGpuCount(struct ncclTopoSystem* system, int* count) {
  *count = system->nodes[GPU].count;
  return ncclSuccess;
}

ncclResult_t ncclTopoGetNetCount(struct ncclTopoSystem* system, int* count) {
  *count = system->nodes[NET].count;
  return ncclSuccess;
}

ncclResult_t ncclTopoGetNvsCount(struct ncclTopoSystem* system, int* count) {
  *count = system->nodes[NVS].count;
  return ncclSuccess;
}

ncclResult_t ncclTopoGetCompCap(struct ncclTopoSystem* system, int* ccMin, int* ccMax) {
  if (system->nodes[GPU].count == 0) return ncclInternalError;
  int min, max;
  min = max = system->nodes[GPU].nodes[0].gpu.cudaCompCap;
  for (int g=1; g<system->nodes[GPU].count; g++) {
    min = std::min(min, system->nodes[GPU].nodes[g].gpu.cudaCompCap);
    max = std::max(max, system->nodes[GPU].nodes[g].gpu.cudaCompCap);
  }
  if (ccMin) *ccMin = min;
  if (ccMax) *ccMax = max;
  return ncclSuccess;
}
