/*************************************************************************
 * Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <ctype.h>
#include <float.h>
#include "core.h"
#include "nvmlwrap.h"
#include "xml.h"
#if defined(__x86_64__)
#include <cpuid.h>
#endif

// Arbitrarily large number for constructing virtual topology string
#define NCCL_MAX_XML_DEPTH 1024

/*******************/
/* XML File Parser */
/*******************/

ncclResult_t xmlGetChar(FILE* file, char* c) {
  if (fread(c, 1, 1, file) == 0) {
    WARN("XML Parse : Unexpected EOF");
    return ncclInternalError;
  }
  return ncclSuccess;
}

ncclResult_t xmlGetValue(FILE* file, char* value, char* last) {
  char c;
  NCCLCHECK(xmlGetChar(file, &c));
  if (c != '"' && c != '\'') {
#if INT_OK
    int o = 0;
    do {
      value[o++] = c;
      NCCLCHECK(xmlGetChar(file, &c));
    } while (c >= '0' && c <= '9');
    value[o] = '\0';
    *last = c;
    return ncclSuccess;
#else
    WARN("XML Parse : Expected (double) quote.");
    return ncclInternalError;
#endif
  }
  int o = 0;
  do {
    NCCLCHECK(xmlGetChar(file, &c));
    value[o++] = c;
  } while (c != '"');
  value[o-1] = '\0';
  NCCLCHECK(xmlGetChar(file, last));
  return ncclSuccess;
}

ncclResult_t xmlGetToken(FILE* file, char* name, char* value, char* last) {
  char c;
  char* ptr = name;
  int o = 0;
  do {
    NCCLCHECK(xmlGetChar(file, &c));
    if (c == '=') {
      ptr[o] = '\0';
      if (value == NULL) {
        WARN("XML Parse : Unexpected value with name %s", ptr);
        return ncclInternalError;
      }
      return xmlGetValue(file, value, last);
    }
    ptr[o] = c;
    if (o == MAX_STR_LEN-1) {
      ptr[o] = '\0';
      WARN("Error : name %s too long (max %d)", ptr, MAX_STR_LEN);
      return ncclInternalError;
    }
    o++;
  } while (c != ' ' && c != '>' && c != '/' && c != '\n' && c != '\r');
  ptr[o-1] = '\0';
  *last = c;
  return ncclSuccess;
}

// Shift the 3-chars string by one char and append c at the end
#define SHIFT_APPEND(s, c) do { s[0]=s[1]; s[1]=s[2]; s[2]=c; } while(0)
ncclResult_t xmlSkipComment(FILE* file, char* start, char next) {
  // Start from something neutral with \0 at the end.
  char end[4] = "...";

  // Inject all trailing chars from previous reads. We don't need
  // to check for --> here because there cannot be a > in the name.
  for (int i=0; i<strlen(start); i++) SHIFT_APPEND(end, start[i]);
  SHIFT_APPEND(end, next);

  // Stop when we find "-->"
  while (strcmp(end, "-->") != 0) {
    int c;
    if (fread(&c, 1, 1, file) != 1) {
      WARN("XML Parse error : unterminated comment");
      return ncclInternalError;
    }
    SHIFT_APPEND(end, c);
  }
  return ncclSuccess;
}

ncclResult_t xmlGetNode(FILE* file, struct ncclXmlNode* node) {
  node->type = NODE_TYPE_NONE;
  char c = ' ';
  while (c == ' ' || c == '\n' || c == '\r') {
    if (fread(&c, 1, 1, file) == 0) return ncclSuccess;
  }
  if (c != '<') {
    WARN("XML Parse error : expecting '<', got '%c'", c);
    return ncclInternalError;
  }
  // Read XML element name
  NCCLCHECK(xmlGetToken(file, node->name, NULL, &c));

  // Check for comments
  if (strncmp(node->name, "!--", 3) == 0) {
    NCCLCHECK(xmlSkipComment(file, node->name+3, c));
    return xmlGetNode(file, node);
  }

  // Check for closing tag
  if (node->name[0] == '\0' && c == '/') {
    node->type = NODE_TYPE_CLOSE;
    // Re-read the name, we got '/' in the first call
    NCCLCHECK(xmlGetToken(file, node->name, NULL, &c));
    if (c != '>') {
      WARN("XML Parse error : unexpected trailing %c in closing tag %s", c, node->name);
      return ncclInternalError;
    }
    return ncclSuccess;
  }

  node->type = NODE_TYPE_OPEN;

  // Get Attributes
  int a = 0;
  while (c == ' ') {
    NCCLCHECK(xmlGetToken(file, node->attrs[a].key, node->attrs[a].value, &c));
    if (a == MAX_ATTR_COUNT) {
      INFO(NCCL_GRAPH, "XML Parse : Ignoring extra attributes (max %d)", MAX_ATTR_COUNT);
      // Actually we need to still consume the extra attributes so we have an extra one.
    } else a++;
  }
  node->nAttrs = a;
  if (c == '/') {
    node->type = NODE_TYPE_SINGLE;
    char str[MAX_STR_LEN];
    NCCLCHECK(xmlGetToken(file, str, NULL, &c));
  }
  if (c != '>') {
    WARN("XML Parse : expected >, got '%c'", c);
    return ncclInternalError;
  }
  return ncclSuccess;
}

typedef ncclResult_t (*xmlHandlerFunc_t)(FILE*, struct ncclXml*, struct ncclXmlNode*);

struct xmlHandler {
  const char * name;
  xmlHandlerFunc_t func;
};
//注意会递归调用
ncclResult_t xmlLoadSub(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head, struct xmlHandler handlers[], int nHandlers) {
  if (head && head->type == NODE_TYPE_SINGLE) return ncclSuccess;
  while (1) {
    if (xml->maxIndex == xml->maxNodes) {
      WARN("Error : XML parser is limited to %d nodes", xml->maxNodes);
      return ncclInternalError;
    }
    struct ncclXmlNode* node = xml->nodes+xml->maxIndex;
    memset(node, 0, sizeof(struct ncclXmlNode));
    NCCLCHECK(xmlGetNode(file, node));
    if (node->type == NODE_TYPE_NONE) {
      if (head) {
        WARN("XML Parse : unterminated %s", head->name);
        return ncclInternalError;
      } else {
        // All done
        return ncclSuccess;
      }
    }
    if (head && node->type == NODE_TYPE_CLOSE) {
      if (strcmp(node->name, head->name) != 0) {
        WARN("XML Mismatch : %s / %s", head->name, node->name);
        return ncclInternalError;
      }
      return ncclSuccess;
    }
    int found = 0;
    for (int h=0; h<nHandlers; h++) {
      if (strcmp(node->name, handlers[h].name) == 0) {
        if (head) {
          if (head->nSubs == MAX_SUBS) {
            WARN("Error : XML parser is limited to %d subnodes", MAX_SUBS);
            return ncclInternalError;
          }
          head->subs[head->nSubs++] = node;
        }
        node->parent = head;
        node->nSubs = 0;
        xml->maxIndex++;
        NCCLCHECK(handlers[h].func(file, xml, node));
        found = 1;
        break;
      }
    }
    if (!found) {
      if (nHandlers) INFO(NCCL_GRAPH, "Ignoring element %s", node->name);
      NCCLCHECK(xmlLoadSub(file, xml, node, NULL, 0));
    }
  }
}

/**************/
/* XML Writer */
/**************/

// exp == 1 -- serialize; exp == 0 -- deserialize
ncclResult_t ncclTopoConvertXml(struct ncclXml* xml, uintptr_t base, int exp) {
  for (int n = 0; n < xml->maxIndex; n++) {
    struct ncclXmlNode *node = &xml->nodes[n];

    // For "parent", we shift the base by 1 so that we can distinguish actual
    // NULL pointers from pointers pointing to the first node.
    if (node->parent)
      node->parent = (struct ncclXmlNode *) (exp ? ((uintptr_t)node->parent - base + 1) : (base - 1 + (uintptr_t)node->parent));

    for (int s = 0; s < node->nSubs; s++) {
      node->subs[s] = (struct ncclXmlNode *) (exp ? ((uintptr_t)node->subs[s] - base) : (base + (uintptr_t)node->subs[s]));
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoDumpXmlRec(int indent, FILE* file, struct ncclXmlNode* node) {
  for (int i=0; i<indent; i++) fprintf(file, " ");
  fprintf(file, "<%s", node->name);

  for (int a=0; a<node->nAttrs; a++) {
    fprintf(file, " %s=\"%s\"", node->attrs[a].key, node->attrs[a].value);
  }
  if (node->nSubs == 0) {
    fprintf(file, "/>\n");
  } else {
    fprintf(file, ">\n");
    for (int s=0; s<node->nSubs; s++) {
      NCCLCHECK(ncclTopoDumpXmlRec(indent+2, file, node->subs[s]));
    }
    for (int i=0; i<indent; i++) fprintf(file, " ");
    fprintf(file, "</%s>\n", node->name);
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoDumpXmlToFile(const char* xmlTopoFile, struct ncclXml* xml) {
  FILE* file = fopen(xmlTopoFile, "w");
  if (file == NULL) {
    WARN("Unable to open %s, not dumping topology.", xmlTopoFile);
    return ncclSuccess;
  }
  NCCLCHECK(ncclTopoDumpXmlRec(0, file, xml->nodes));
  fclose(file);
  return ncclSuccess;
}
//用于将 srcParent （源 XML 子树）融合到 dstParent （目标 XML 子树）下。
static ncclResult_t xmlTopoFuseXmlRecursive(struct ncclXml* dst, struct ncclXmlNode* dstParent, struct ncclXmlNode* srcParent) {
  for (int i = 0; i < srcParent->nSubs; i++) {
    struct ncclXmlNode* srcNode = srcParent->subs[i];
    struct ncclXmlNode* dstNode;
    NCCLCHECK(xmlFindNode(dstParent, srcNode, &dstNode));//在目标树的当前父节点下查找是否已经有与 srcNode 相同的节点（一般是通过节点名和关键属性判断）。
    if (dstNode == NULL) {
      NCCLCHECK(xmlAddTree(dst, dstParent, srcNode));//直接把 srcNode 整棵子树挂到 dstParent(常见的就是system) 下（ xmlAddTree ）
    } else {
      // 如果找到了（ dstNode != NULL ），说明目标树已经有同名节点，则递归调用自己，把 srcNode 的子树继续融合到 dstNode 下
      NCCLCHECK(xmlTopoFuseXmlRecursive(dst, dstNode, srcNode));
    }
  }
  return ncclSuccess;
}
// 作用是把 src （源 XML）融合到 dst （目标 XML）中。
ncclResult_t ncclTopoFuseXml(struct ncclXml* dst, struct ncclXml* src) {
  struct ncclXmlNode* topNodeDst;
  NCCLCHECK(xmlFindTag(dst, "system", &topNodeDst));
  //如果目标 XML 没有 <system> 节点，直接把源 XML 的根节点整个加进来（ xmlAddTree ），融合结束。
  if (topNodeDst == NULL) {
    xmlAddTree(dst, NULL, src->nodes);
    return ncclSuccess;
  }
  // 如果目标 XML 已经有 <system> 节点，则在源 XML 里也找 <system> 节点，然后调用递归融合函数 xmlTopoFuseXmlRecursive 
  // ，把源 <system> 节点下的内容融合到目标 <system> 节点下。
  struct ncclXmlNode* topNodeSrc;
  NCCLCHECK(xmlFindTag(src, "system", &topNodeSrc));

  NCCLCHECK(xmlTopoFuseXmlRecursive(dst, topNodeDst, topNodeSrc));

  return ncclSuccess;
}


/****************************************/
/* Parser rules for our specific format */
/****************************************/

ncclResult_t ncclTopoXmlLoadNvlink(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  NCCLCHECK(xmlLoadSub(file, xml, head, NULL, 0));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadPciLink(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  NCCLCHECK(xmlLoadSub(file, xml, head, NULL, 0));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadC2c(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  NCCLCHECK(xmlLoadSub(file, xml, head, NULL, 0));
  return ncclSuccess;
}
ncclResult_t ncclTopoXmlLoadGpu(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  struct xmlHandler handlers[] = { { "nvlink", ncclTopoXmlLoadNvlink }, { "c2c", ncclTopoXmlLoadC2c } };
  NCCLCHECK(xmlLoadSub(file, xml, head, handlers, 2));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadNet(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  NCCLCHECK(xmlLoadSub(file, xml, head, NULL, 0));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadNic(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  struct xmlHandler handlers[] = { { "net", ncclTopoXmlLoadNet } };
  NCCLCHECK(xmlLoadSub(file, xml, head, handlers, 1));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadPci(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  struct xmlHandler handlers[] = { { "pci", ncclTopoXmlLoadPci }, { "gpu", ncclTopoXmlLoadGpu }, { "nic", ncclTopoXmlLoadNic}, { "pcilink", ncclTopoXmlLoadPciLink} };
  NCCLCHECK(xmlLoadSub(file, xml, head, handlers, 4));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadCpu(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  struct xmlHandler handlers[] = { { "pci", ncclTopoXmlLoadPci }, { "nic", ncclTopoXmlLoadNic } };
  NCCLCHECK(xmlLoadSub(file, xml, head, handlers, 2));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadSystem(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  int version;
  NCCLCHECK(xmlGetAttrInt(head, "version", &version));
  if (version != NCCL_TOPO_XML_VERSION) {
    WARN("XML Topology has wrong version %d, %d needed", version, NCCL_TOPO_XML_VERSION);
    return ncclInvalidUsage;
  }
  const char* name;
  NCCLCHECK(xmlGetAttr(head, "name", &name));
  if (name != NULL) INFO(NCCL_GRAPH, "Loading topology %s", name);
  else INFO(NCCL_GRAPH, "Loading unnamed topology");

  struct xmlHandler handlers[] = { { "cpu", ncclTopoXmlLoadCpu } };
  NCCLCHECK(xmlLoadSub(file, xml, head, handlers, 1));
  return ncclSuccess;
}

ncclResult_t ncclTopoGetXmlFromFile(const char* xmlTopoFile, struct ncclXml* xml, int warn) {
  FILE* file = fopen(xmlTopoFile, "r");
  if (file == NULL) {
    if (warn) {
      WARN("Could not open XML topology file %s : %s", xmlTopoFile, strerror(errno));
    }
    return ncclSuccess;
  }
  INFO(NCCL_GRAPH, "Loading topology file %s", xmlTopoFile);
  struct xmlHandler handlers[] = { { "system", ncclTopoXmlLoadSystem } };
  xml->maxIndex = 0;
  NCCLCHECK(xmlLoadSub(file, xml, NULL, handlers, 1));
  fclose(file);
  return ncclSuccess;
}

/**********************/
/* XML creation       */
/* from autodetection */
/**********************/

#define BUSID_SIZE (sizeof("0000:00:00.0"))
#define BUSID_REDUCED_SIZE (sizeof("0000:00"))
static void memcpylower(char* dst, const char* src, const size_t size) {
  for (int i=0; i<size; i++) dst[i] = tolower(src[i]);
}
static ncclResult_t getPciPath(const char* busId, char** path) {
  char busPath[] = "/sys/class/pci_bus/0000:00/../../0000:00:00.0";
  memcpylower(busPath+sizeof("/sys/class/pci_bus/")-1, busId, BUSID_REDUCED_SIZE-1);
  memcpylower(busPath+sizeof("/sys/class/pci_bus/0000:00/../../")-1, busId, BUSID_SIZE-1);
  *path = realpath(busPath, NULL);
  if (*path == NULL) {
    WARN("Could not find real path of %s", busPath);
    return ncclSystemError;
  }
  return ncclSuccess;
}

#include <dirent.h>
static ncclResult_t getBcmLinks(const char* busId, int* nlinks, char** peers) {
  *nlinks = 0;
  *peers = NULL;
  char dirPath[] = "/sys/kernel/pci_switch_link/virtual_switch_links/0000:00:00.0";
  memcpylower(dirPath+sizeof("/sys/kernel/pci_switch_link/virtual_switch_links/")-1, busId, BUSID_SIZE-1);
  DIR *dir = opendir(dirPath);
  if (dir) {
    struct dirent* file;
    while ((file = readdir(dir)) != NULL) {
      if (strlen(file->d_name) != BUSID_SIZE-1) continue;
      char* path;
      if (getPciPath(file->d_name, &path) == ncclSystemError) continue;
      free(path);
      NCCLCHECK(ncclRealloc(peers, (*nlinks)*BUSID_SIZE, ((*nlinks)+1)*BUSID_SIZE));
      memcpy((*peers)+BUSID_SIZE*(*nlinks)++, file->d_name, BUSID_SIZE);
    }
    closedir(dir);
  }
  return ncclSuccess;
}
//根据路径读取文件内容，获取对应的值
ncclResult_t ncclTopoGetStrFromSys(const char* path, const char* fileName, char* strValue) {
  char filePath[PATH_MAX];
  snprintf(filePath, sizeof(filePath), "%s/%s", path, fileName);
  int offset = 0;
  FILE* file;
  if ((file = fopen(filePath, "r")) != NULL) {
    while (feof(file) == 0 && ferror(file) == 0 && offset < MAX_STR_LEN) {
      int len = fread(strValue+offset, 1, MAX_STR_LEN-offset, file);
      offset += len;
    }
    fclose(file);
  }
  if (offset == 0) {
    strValue[0] = '\0';
    INFO(NCCL_GRAPH, "Topology detection : could not read %s, ignoring", filePath);
  } else {
    strValue[offset-1] = '\0';
  }
  return ncclSuccess;
}
//从指定的 sysfs 路径读取文件内容，并将其作为属性值设置到 XML 节点上 。 filename:要读取的文件名（如 "vendor" 、 "device"）。
//path ：sysfs 路径（如 /sys/class/pci_bus/0000:65/../../0000:65:00.0
ncclResult_t ncclTopoSetAttrFromSys(struct ncclXmlNode* pciNode, const char* path, const char* fileName, const char* attrName) {
  char strValue[MAX_STR_LEN];
  NCCLCHECK(ncclTopoGetStrFromSys(path, fileName, strValue));//读取文件内容，获取属性值
  if (strValue[0] != '\0') { NCCLCHECK(xmlSetAttr(pciNode, attrName, strValue)); }//将其设置为 XML 节点的属性
  TRACE(NCCL_GRAPH, "Read from sys %s/%s -> %s=%s", path, fileName, attrName, strValue);
  return ncclSuccess;
}

ncclResult_t ncclTopoGetXmlFromCpu(struct ncclXmlNode* cpuNode, struct ncclXml* xml) {
  int index;
  NCCLCHECK(xmlGetAttrIndex(cpuNode, "affinity", &index));
  if (index == -1) {
    const char* numaId;
    NCCLCHECK(xmlGetAttr(cpuNode, "numaid", &numaId));
    if (numaId == NULL) {
      WARN("GetXmlFromCpu : could not find CPU numa ID.");
      return ncclInternalError;
    }
    // Set affinity
    char cpumaskPath[] = "/sys/devices/system/node/node000000";
    snprintf(cpumaskPath, sizeof(cpumaskPath), "/sys/devices/system/node/node%s", numaId);
    NCCLCHECK(ncclTopoSetAttrFromSys(cpuNode, cpumaskPath, "cpumap", "affinity"));
  }

  NCCLCHECK(xmlGetAttrIndex(cpuNode, "arch", &index));
  if (index == -1) {
    // Fill CPU type / vendor / model
#if defined(__PPC__)
    NCCLCHECK(xmlSetAttr(cpuNode, "arch", "ppc64"));
#elif defined(__aarch64__)
    NCCLCHECK(xmlSetAttr(cpuNode, "arch", "arm64"));
#elif defined(__x86_64__)
    NCCLCHECK(xmlSetAttr(cpuNode, "arch", "x86_64"));
#endif
  }

#if defined(__x86_64__)
  NCCLCHECK(xmlGetAttrIndex(cpuNode, "vendor", &index));
  if (index == -1) {
    union {
      struct {
        // CPUID 0 String register order
        uint32_t ebx;
        uint32_t edx;
        uint32_t ecx;
      };
      char vendor[12];
    } cpuid0;

    unsigned unused;
    __cpuid(0, unused, cpuid0.ebx, cpuid0.ecx, cpuid0.edx);
    char vendor[13];
    strncpy(vendor, cpuid0.vendor, 12);
    vendor[12] = '\0';
    NCCLCHECK(xmlSetAttr(cpuNode, "vendor", vendor));
  }

  NCCLCHECK(xmlGetAttrIndex(cpuNode, "familyid", &index));
  if (index == -1) {
    union {
      struct {
        unsigned steppingId:4;
        unsigned modelId:4;
        unsigned familyId:4;
        unsigned processorType:2;
        unsigned resv0:2;
        unsigned extModelId:4;
        unsigned extFamilyId:8;
        unsigned resv1:4;
      };
      uint32_t val;
    } cpuid1;
    unsigned unused;
    __cpuid(1, cpuid1.val, unused, unused, unused);
    int familyId = cpuid1.familyId + (cpuid1.extFamilyId << 4);
    int modelId = cpuid1.modelId + (cpuid1.extModelId << 4);
    NCCLCHECK(xmlSetAttrInt(cpuNode, "familyid", familyId));
    NCCLCHECK(xmlSetAttrInt(cpuNode, "modelid", modelId));
  }
#endif
  return ncclSuccess;
}
//在 XML 拓扑结构中查找指定 busId 的 PCI 节点，如果没有则新建一个，并返回该节点指针
ncclResult_t ncclTopoGetPciNode(struct ncclXml* xml, const char* busId, struct ncclXmlNode** pciNode) {
  NCCLCHECK(xmlFindTagKv(xml, "pci", pciNode, "busid", busId));
  if (*pciNode == NULL) {
    NCCLCHECK(xmlAddNode(xml, NULL, "pci", pciNode));
    NCCLCHECK(xmlSetAttr(*pciNode, "busid", busId));
  }
  return ncclSuccess;
}

// Check whether a string is in BDF format or not.
// BDF (Bus-Device-Function) is "BBBB:BB:DD.F" where B, D and F are hex digits.
// There can be trailing chars.
int isHex(char c) { return ((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')); }
int checkBDFFormat(char* bdf) {
  if (strlen(bdf) != 12) return 0;
  if ((bdf[4] != ':') || (bdf[7] != ':') || (bdf[10] != '.')) return 0;
  if ((isHex(bdf[0]) == 0) || (isHex(bdf[1]) == 0) || (isHex(bdf[2]) == 0) || (isHex(bdf[3]) == 0) ||
      (isHex(bdf[5]) == 0) || (isHex(bdf[6]) == 0) || (isHex(bdf[8]) == 0) || (isHex(bdf[9]) == 0) ||
      (isHex(bdf[11]) == 0)) return 0;
  return 1;
}
//根据 PCI 节点的 busId，从系统 /sys 路径自动补全该 PCI 设备的详细属性，并自动挂载到正确的父节点（如上级 PCI 交换机或 CPU） 。
ncclResult_t ncclTopoGetXmlFromSys(struct ncclXmlNode* pciNode, struct ncclXml* xml) {
  // Fill info, then parent
  const char* busId;
  NCCLCHECK(xmlGetAttr(pciNode, "busid", &busId));
  char* path = NULL;
  ncclDebugNoWarn = NCCL_GRAPH;
  getPciPath(busId, &path);//获取该 busId 在 /sys 下的真实路径
  ncclDebugNoWarn = 0;
  //如果能找到 sysfs 路径，则依次尝试补全 class 、 vendor 、 device 、 subsystem_vendor 、 subsystem_device 等属性（如果还没有）
  if (path) {
    NCCLCHECK(ncclTopoSetAttrFromSys(pciNode, path, "class", "class"));
  }
  int index;
  ncclDebugNoWarn = NCCL_GRAPH;
  NCCLCHECK(xmlGetAttrIndex(pciNode, "vendor", &index));//先测试一下有没有 vendor 属性。其他也同理
  if (index == -1) {
    if (path) ncclTopoSetAttrFromSys(pciNode, path, "vendor", "vendor");
  }
  NCCLCHECK(xmlGetAttrIndex(pciNode, "device", &index));
  if (index == -1) {
    if (path) ncclTopoSetAttrFromSys(pciNode, path, "device", "device");
  }
  NCCLCHECK(xmlGetAttrIndex(pciNode, "subsystem_vendor", &index));
  if (index == -1) {
    if (path) ncclTopoSetAttrFromSys(pciNode, path, "subsystem_vendor", "subsystem_vendor");
  }
  NCCLCHECK(xmlGetAttrIndex(pciNode, "subsystem_device", &index));
  if (index == -1) {
    if (path) ncclTopoSetAttrFromSys(pciNode, path, "subsystem_device", "subsystem_device");
  }
  ncclDebugNoWarn = 0;
  NCCLCHECK(xmlGetAttrIndex(pciNode, "link_speed", &index));
  //补全 link_speed 和 link_width ，会比较 device 和 port 的值，取较小者。
  if (index == -1) {
    if (path) {
      char deviceSpeedStr[MAX_STR_LEN];
      float deviceSpeed = FLT_MAX;
      //链路速率，表示 PCIe 通道每秒传输的数据速率，通常以“GT/s”为单位。决定了单条 PCIe 通道的理论最大带宽。实际带宽还要乘以 link_width ，才能得到实际的带宽。
      
      NCCLCHECK(ncclTopoGetStrFromSys(path, "max_link_speed", deviceSpeedStr));//获取max_link_speed
      sscanf(deviceSpeedStr, "%f GT/s", &deviceSpeed);
      char portSpeedStr[MAX_STR_LEN];
      float portSpeed = FLT_MAX;
      //portSpeedStr 表示上游端口（如主板插槽、PCIe交换机等）的最大链路速率
      NCCLCHECK(ncclTopoGetStrFromSys(path, "../max_link_speed", portSpeedStr));
      sscanf(portSpeedStr, "%f GT/s", &portSpeed);
      //选最小的设置，实际链路速率只能达到两端中较低的那个速率
      NCCLCHECK(xmlSetAttr(pciNode, "link_speed", portSpeed < deviceSpeed ? portSpeedStr : deviceSpeedStr));
    } else {
      NCCLCHECK(xmlSetAttr(pciNode, "link_speed", ""));
    }
  }
  NCCLCHECK(xmlGetAttrIndex(pciNode, "link_width", &index));//链路宽度，表示 PCIe 通道的数量（即 x1、x4、x8、x16 等）
  if (index == -1) {
    if (path) {
      char strValue[MAX_STR_LEN];
      NCCLCHECK(ncclTopoGetStrFromSys(path, "max_link_width", strValue));
      int deviceWidth = strtol(strValue, NULL, 0);
      NCCLCHECK(ncclTopoGetStrFromSys(path, "../max_link_width", strValue));
      int portWidth = strtol(strValue, NULL, 0);
      NCCLCHECK(xmlSetAttrInt(pciNode, "link_width", std::min(deviceWidth,portWidth)));
    } else {
      NCCLCHECK(xmlSetAttr(pciNode, "link_width", ""));
    }
  }

  const char* vendor;
  NCCLCHECK(xmlGetAttr(pciNode, "vendor", &vendor));
  //当检测到当前 PCI 设备的 vendor 是 "0x1000"（即 Broadcom BCM PCIe 交换机）时，自动查找并补全该交换机的 P2P（Peer-to-Peer）连接信息。
  //博通的这个交换机可以理解为nvSwitch的pcie版
  //自动补全 PCIe 交换机的 P2P 拓扑信息 ，让 NCCL 能够正确识别和利用 PCIe 交换机下的直连路径，优化 GPU 间通信。
  if (vendor != NULL && strcmp(vendor, "0x1000") == 0) { // BCM switch, look for P2P connections
    int nlinks;
    char* peers;
    //调用 getBcmLinks(busId, &nlinks, &peers) ，获取该交换机所有直连的 peer 设备 busId 列表（即 P2P 连接）。
    NCCLCHECK(getBcmLinks(busId, &nlinks, &peers));
    for (int l=0; l<nlinks; l++) {//为每个 peer 创建/补全 pcilink 子节点
      /*
      - 遍历所有 peer busId。
      - 检查当前 pciNode 是否已经有指向该 peer 的 pcilink 子节点（通过 target 属性判断）。
      - 如果没有，则新建一个 pcilink 子节点，并设置其 target 属性为 peer 的 busId。
      */
      char* target = peers+l*BUSID_SIZE;
      struct ncclXmlNode* linkNode;
      NCCLCHECK(xmlGetSubKv(pciNode, "pcilink", &linkNode, "target", target));//寻找有没有这个子节点
      if (linkNode == NULL) {//没找到就自己赋值
        NCCLCHECK(xmlAddNode(xml, pciNode, "pcilink", &linkNode));
        NCCLCHECK(xmlSetAttr(linkNode, "target", target));
      }
    }
  }
/*
  为当前的 PCI 设备节点（pciNode）自动查找并设置其父节点（parent） ，
  即确定它在拓扑结构中的上级是谁（可能是上级 PCIe 交换机，也可能是 CPU/NUMA 节点）。
  这样可以自动构建出完整的 PCIe/CPU 拓扑树。
  在 NCCL 拓扑建模中，CPU 节点其实就是 NUMA 节点，每个 NUMA 节点下挂载着本地的 PCIe 设备（如 GPU、NIC 等）。
  path为：/sys/bus/pci/devices/0000:a5:08.5/类似的形式。
  PCI 桥是连接两个 PCI 总线段的桥接芯片。它可以扩展 PCI 总线的层级结构，把主干总线分成多个分支。PCI 桥本身也是一个 PCI 设备，有自己的 BDF 编号
  PCI 根桥（Root Complex） ： 指的是 CPU（或主板芯片组）和 PCIe 总线之间的桥接单元。它是整个 PCIe 拓扑的起点，负责把 CPU/内存和 PCIe 设备连接起来。
  在 sysfs 路径中通常表现为 pci0000:60 这样的目录名（不是标准的 BDF 格式）
  */
  struct ncclXmlNode* parent = pciNode->parent;
  if (parent == NULL) {
    if (path) {
      // Save that for later in case next step is a CPU
      char numaIdStr[MAX_STR_LEN];
      //先读取该设备的 NUMA 节点编号（numa_node 文件），保存到 numaIdStr
      NCCLCHECK(ncclTopoGetStrFromSys(path, "numa_node", numaIdStr));

      // Go up one level in the PCI tree. Rewind two "/" and follow the upper PCI
      // switch, or stop if we reach a CPU root complex.
      /*
        然后向上遍历 sysfs 路径（即 PCIe 树），每遇到一个 / 就计数（slashCount++），并将该处截断（path[parentOffset] = '\0'）。
        对于每一级路径，判断其是否为 BDF 格式（即 "BBBB:BB:DD.F"），如果不是，说明已经到达 CPU 根节点（Root Complex）：在 XML 拓扑中查找或新建对应 NUMA id 的 CPU 节点，并将其作为 parent。
        如果遇到第二个 / ，则认为到达了上一级 PCIe 交换机：查找或新建对应 busid 的 PCI 节点，并将其作为 parent。
        一旦找到 parent，跳出循环。

        */
      int slashCount = 0;
      int parentOffset;
      for (parentOffset = strlen(path)-1; parentOffset>0; parentOffset--) {
        if (path[parentOffset] == '/') {
          slashCount++;
          path[parentOffset] = '\0';
          int start = parentOffset - 1;
          while (start>0 && path[start] != '/') start--;
          // Check whether the parent path looks like "BBBB:BB:DD.F" or not.
          if (checkBDFFormat(path+start+1) == 0) {//说明已经到达 CPU 根节点（Root Complex）
            // This a CPU root complex. Create a CPU tag and stop there.
            struct ncclXmlNode* topNode;
            NCCLCHECK(xmlFindTag(xml, "system", &topNode));//找到顶端节点。
            NCCLCHECK(xmlGetSubKv(topNode, "cpu", &parent, "numaid", numaIdStr));//从顶端节点中查找有没有cpu这个子节点，满足numaid要求的。如果找到，就赋值给parent
            if (parent == NULL) {//说明没找到cpu，那么就自己新建一个cpu节点。
              NCCLCHECK(xmlAddNode(xml, topNode, "cpu", &parent));
              NCCLCHECK(xmlSetAttrLong(parent, "host_hash", getHostHash()));
              NCCLCHECK(xmlSetAttr(parent, "numaid", numaIdStr));
            }
          } else if (slashCount == 2) {
            // Continue on the upper PCI switch 因为pci桥的可能性不大，所以不考虑。
            for (int i = strlen(path)-1; i>0; i--) {
              if (path[i] == '/') {
                NCCLCHECK(xmlFindTagKv(xml, "pci", &parent, "busid", path+i+1));//没找到则添加这个pci Switch。
                if (parent == NULL) {
                  NCCLCHECK(xmlAddNode(xml, NULL, "pci", &parent));
                  NCCLCHECK(xmlSetAttr(parent, "busid", path+i+1));
                }
                break;
              }
            }
          }
        }
        if (parent) break;
      }
    } else {//当无法通过文件系统获取到path路径的时候
      /*
        当 NCCL 无法通过系统信息确定 PCI 设备（如 GPU）属于哪个 CPU/NUMA 节点时，会自动在拓扑中创建一个 numaid=-1 的“未知 CPU 节点”，并将该设备挂载到这个节点下。
        这样可以保证拓扑结构的完整性，即使部分硬件信息缺失，也不会导致拓扑断裂或解析失败。
        这里解释一下：如果 getPciPath 失败，说明系统里没有这个 busid 对应的 PCI 设备，或者 sysfs 信息不全，导致无法定位它在拓扑中的准确位置。
      NCCL 的逻辑是：只有拿到 path，才能通过分析 sysfs 目录结构，找到上级 PCI 设备或 CPU/NUMA 节点。如果 path 为 null，说明这条链断了，无法再通过系统信息推断父节点。
      所以只能把这个设备挂到“未知 CPU/NUMA 节点”下 。这就是为什么会新建一个 numaid=-1 的 cpu 节点，把设备挂在这里。这样做是为了保证 XML 拓扑结构的完整性，即使有设备无法被系统识别，也不会导致拓扑断裂。
      这是一个兜底逻辑。  */
      // No information on /sys, attach GPU to unknown CPU 查找 numaid=-1 的 CPU 节点。 numaid=-1 表示“未知 NUMA 节点”，即系统无法确定该设备属于哪个 NUMA 域。
      NCCLCHECK(xmlFindTagKv(xml, "cpu", &parent, "numaid", "-1"));
      if (parent == NULL) {
        struct ncclXmlNode* topNode;//如果没有找到 numaid=-1 的 CPU 节点，则新建一个。
        NCCLCHECK(xmlFindTag(xml, "system", &topNode));//找到 XML 拓扑的顶层 system 节点，准备在其下添加新的 CPU 节点。
        NCCLCHECK(xmlAddNode(xml, topNode, "cpu", &parent));
        NCCLCHECK(xmlSetAttrLong(parent, "host_hash", getHostHash()));//给新建的 CPU 节点设置 host_hash 属性， getHostHash() 用于唯一标识主机
        NCCLCHECK(xmlSetAttr(parent, "numaid", "-1"));
        NCCLCHECK(ncclTopoGetXmlFromCpu(parent, xml));//为该 CPU 节点补全其它属性（如架构、厂商等）。
      }
    }
    //设置当前pci设备的父节点（pci switch、CPU、或者挂载的cpu）
    pciNode->parent = parent;
    // Keep PCI sub devices ordered by PCI Bus ID (Issue #820)
    // Coverity complains about dereferenced parent being NULL
    // but this can never happen.
    // coverity[var_deref_op]
    int subIndex = parent->nSubs;
    const char* newBusId;
    NCCLCHECK(xmlGetAttrStr(pciNode, "busid", &newBusId));
    //遍历父节点已有的所有子节点，找到第一个 busid 大于当前节点 busid 的位置 subIndex。这样可以保证子节点按照 busid 升序排列。
    for (int s=0; s<parent->nSubs; s++) {
      const char* busId;
      NCCLCHECK(xmlGetAttr(parent->subs[s], "busid", &busId));
      if (busId != NULL && strcmp(newBusId, busId) < 0) { subIndex = s; break; }
    }
    if (parent->nSubs == MAX_SUBS) {
      WARN("Error : XML parser is limited to %d subnodes", MAX_SUBS);
      return ncclInternalError;
    }
    //将后面的子节点依次向后移动，为新节点腾出空间。
    for (int s = parent->nSubs; s > subIndex; s--) parent->subs[s] = parent->subs[s-1];
    parent->subs[subIndex] = pciNode;
    parent->nSubs++;
  }
  //根据父节点的类型，自动补全父节点的属性信息。前面的逻辑中有的补全了父节点的信息，有的没补全，这里进行最终的补全。
  if (strcmp(parent->name, "pci") == 0) {
    NCCLCHECK(ncclTopoGetXmlFromSys(parent, xml));
  } else if (strcmp(parent->name, "cpu") == 0) {
    NCCLCHECK(ncclTopoGetXmlFromCpu(parent, xml));
  }
  free(path);
  return ncclSuccess;
}

ncclResult_t ncclTopoGetXmlFromGpu(struct ncclXmlNode* pciNode, nvmlDevice_t nvmlDev, struct ncclXml* xml, struct ncclXmlNode** gpuNodeRet) {
  struct ncclXmlNode* gpuNode = NULL;
  NCCLCHECK(xmlGetSub(pciNode, "gpu", &gpuNode));
  if (gpuNode == NULL) NCCLCHECK(xmlAddNode(xml, pciNode, "gpu", &gpuNode));

  int index = -1;

  int dev = -1;
  NCCLCHECK(xmlGetAttrIndex(gpuNode, "dev", &index));
  if (index == -1) {
    NCCLCHECK(ncclNvmlDeviceGetIndex(nvmlDev, (unsigned int*)&dev));
    NCCLCHECK(xmlSetAttrInt(gpuNode, "dev", dev));
  }
  NCCLCHECK(xmlGetAttrInt(gpuNode, "dev", &dev));
  if (dev == -1) { *gpuNodeRet = NULL; return ncclSuccess; }

  NCCLCHECK(xmlGetAttrIndex(gpuNode, "sm", &index));
  if (index == -1) {
    int cudaMajor, cudaMinor;
    if (nvmlDev == NULL) {
      cudaDeviceProp devProp;
      CUDACHECK(cudaGetDeviceProperties(&devProp, dev));
      cudaMajor = devProp.major; cudaMinor = devProp.minor;
    } else {
      NCCLCHECK(ncclNvmlDeviceGetCudaComputeCapability(nvmlDev, &cudaMajor, &cudaMinor));
    }
    NCCLCHECK(xmlSetAttrInt(gpuNode, "sm", cudaMajor*10+cudaMinor));
  }
  int sm;
  NCCLCHECK(xmlGetAttrInt(gpuNode, "sm", &sm));

  struct ncclXmlNode* nvlNode = NULL;
  NCCLCHECK(xmlGetSub(gpuNode, "nvlink", &nvlNode));
  if (nvlNode == NULL) {
    // NVML NVLink detection
    int maxNvLinks = (sm < 60) ? 0 : (sm < 70) ? 4 : (sm < 80) ? 6 : (sm < 90) ? 12 : 18;

    if (maxNvLinks > 0 && nvmlDev == NULL) {
      WARN("No NVML device handle. Skipping nvlink detection.");
      maxNvLinks = 0;
    }

    for (int l=0; l<maxNvLinks; ++l) {
      // Check whether we can use this NVLink for P2P
      unsigned canP2P;
      if ((ncclNvmlDeviceGetNvLinkCapability(nvmlDev, l, NVML_NVLINK_CAP_P2P_SUPPORTED, &canP2P) != ncclSuccess) || !canP2P) continue;

      // Make sure the Nvlink is up. The previous call should have trained the link.
      nvmlEnableState_t isActive = NVML_FEATURE_DISABLED;
#if CUDART_VERSION >= 11080
      if (sm >= 90) {
        nvmlFieldValue_t fv;
        fv.fieldId = NVML_FI_DEV_NVLINK_GET_STATE;
        fv.scopeId = l;
        // fv.value will contain NV_FEATURE_ENABLED or NV_FEATURE_DISABLED
        if ((ncclNvmlDeviceGetFieldValues(nvmlDev, 1, &fv) == ncclSuccess) && (fv.nvmlReturn == NVML_SUCCESS))
          isActive = (nvmlEnableState_t) fv.value.uiVal;
      } else /* FALLTHRU to GetNvLinkState if before SM90 */
#endif
      {
        (void) ncclNvmlDeviceGetNvLinkState(nvmlDev, l, &isActive);
      }
      if (isActive != NVML_FEATURE_ENABLED) continue;

      // Try to figure out what's on the other side of the NVLink
      nvmlPciInfo_t remoteProc;
      if (ncclNvmlDeviceGetNvLinkRemotePciInfo(nvmlDev, l, &remoteProc) != ncclSuccess) continue;

      // Make a lower case copy of the bus ID for calling ncclDeviceType
      // PCI system path is in lower case
      char* p = remoteProc.busId;
      char lowerId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
      for (int c=0; c<NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE; c++) {
        lowerId[c] = tolower(p[c]);
        if (p[c] == 0) break;
      }

      NCCLCHECK(xmlGetSubKv(gpuNode, "nvlink", &nvlNode, "target", lowerId));
      if (nvlNode == NULL) {
        NCCLCHECK(xmlAddNode(xml, gpuNode, "nvlink", &nvlNode));
        NCCLCHECK(xmlSetAttr(nvlNode, "target", lowerId));
        NCCLCHECK(xmlSetAttrInt(nvlNode, "count", 1));
      } else {
        int count;
        NCCLCHECK(xmlGetAttrInt(nvlNode, "count", &count));
        NCCLCHECK(xmlSetAttrInt(nvlNode, "count", count+1));
      }
    }
  }
#if CUDART_VERSION >= 11080
  struct ncclXmlNode* c2cNode = NULL;
  NCCLCHECK(xmlGetSub(gpuNode, "c2c", &c2cNode));
  if (c2cNode == NULL) {
      if (sm >= 90) {
        int c2cLinksCount = 0;
        nvmlFieldValue_t fv;
        fv.fieldId = NVML_FI_DEV_C2C_LINK_COUNT;
        if ((ncclNvmlDeviceGetFieldValues(nvmlDev, 1, &fv) == ncclSuccess) && (fv.nvmlReturn == NVML_SUCCESS)) {
          c2cLinksCount = fv.value.uiVal;
          int bw = 0;
	  int count = 0;
          for (int l=0; l<c2cLinksCount; l++) {
            nvmlFieldValue_t fvs[2];
            fvs[0].fieldId = NVML_FI_DEV_C2C_LINK_GET_STATUS;
            fvs[0].scopeId = l;
            fvs[1].fieldId = NVML_FI_DEV_C2C_LINK_GET_MAX_BW;
            fvs[1].scopeId = l;
            if ((ncclNvmlDeviceGetFieldValues(nvmlDev, 2, fvs) == ncclSuccess) &&
                (fvs[0].nvmlReturn == NVML_SUCCESS) &&
                (fvs[0].value.uiVal == 1) &&
                (fvs[1].nvmlReturn == NVML_SUCCESS)) {
              bw = fvs[1].value.uiVal;
	      count++;
            }
          }
          if (count > 0) {
            NCCLCHECK(xmlAddNode(xml, gpuNode, "c2c", &c2cNode));
            NCCLCHECK(xmlSetAttrInt(c2cNode, "bw", bw));
            NCCLCHECK(xmlSetAttrInt(c2cNode, "count", count));
          }
        }
      }
  }
#endif
  // Fill target classes
  for (int s=0; s<gpuNode->nSubs; s++) {
    struct ncclXmlNode* sub = gpuNode->subs[s];
    if (strcmp(sub->name, "nvlink") != 0) continue;
    int index;
    NCCLCHECK(xmlGetAttrIndex(sub, "tclass", &index));
    if (index == -1) {
      const char* busId;
      NCCLCHECK(xmlGetAttr(sub, "target", &busId));
      char* path;
      ncclDebugNoWarn = NCCL_GRAPH;
      getPciPath(busId, &path);
      ncclDebugNoWarn = 0;
      if (path == NULL || strcmp(busId, "fffffff:ffff:ff") == 0) {
        // Remote NVLink device is not visible inside this VM. Assume NVSwitch.
        NCCLCHECK(xmlSetAttr(sub, "tclass", "0x068000"));
      } else {
        NCCLCHECK(ncclTopoSetAttrFromSys(sub, path, "class", "tclass"));
        free(path);
      }
    }
  }
  *gpuNodeRet = gpuNode;
  return ncclSuccess;
}

ncclResult_t ncclTopoFillGpu(struct ncclXml* xml, const char* busId, struct ncclXmlNode** gpuNode) {
  struct ncclXmlNode* node;
  NCCLCHECK(ncclTopoGetPciNode(xml, busId, &node));//在 XML 拓扑结构中查找指定 busId 的 PCI 节点，没有找到就新建一个并让node指向它。
  NCCLCHECK(xmlSetAttrIfUnset(node, "class", "0x03"));//如果该节点还没有 class 属性，则设置为 "0x03" ，表示这是一个显示控制器（GPU）的 PCI class code。
  NCCLCHECK(ncclTopoGetXmlFromSys(node, xml));//自动从 /sys 文件系统中读取该 PCI 设备的详细信息（如 vendor、device、link_speed 等），并补全到 XML 节点属性中。并自动挂载到正确的父节点（如上级 PCI 交换机或 CPU） 。
  nvmlDevice_t nvmlDev;
  NCCLCHECK(ncclNvmlDeviceGetHandleByPciBusId(busId, &nvmlDev));//通过 PCI Bus ID（ busId ）获取对应的 GPU 设备句柄。
  //利用刚刚获得的 nvmlDev 设备句柄，结合当前 XML 节点 node 、XML 结构 xml 和 gpuNode ，补全该 GPU 节点的详细属性信息。
  //这里的node相当于pci节点，而gpu是挂载在其下面的。
  NCCLCHECK(ncclTopoGetXmlFromGpu(node, nvmlDev, xml, gpuNode));
  return ncclSuccess;
}

// Returns the subsystem name of a path, i.e. the end of the path
// where sysPath/subsystem points to.
/*
这个函数的作用是：给定一个 sysfs 路径（如 /sys/class/net/eth0 ），
返回该路径下 subsystem 软链接所指向的子系统名称（比如 pci 、 usb 、 virtual、net 等），并将其写入 subSys 字符串。
假设 sysPath 是 /sys/class/net/eth0 ， /sys/class/net/eth0/subsystem 可能是一个软链接，指向 /sys/class/net 或 /sys/class/pci 等。
*/
ncclResult_t ncclTopoGetSubsystem(const char* sysPath, char* subSys) {
  char subSysPath[PATH_MAX];
  snprintf(subSysPath, sizeof(subSysPath), "%s/subsystem", sysPath);
  char* path = realpath(subSysPath, NULL);//获得其真实路径（即软链接指向的实际路径）。
  if (path == NULL) {
    subSys[0] = '\0';
  } else {
    int offset;
    for (offset = strlen(path); offset > 0 && path[offset] != '/'; offset--);
    strcpy(subSys, path+offset+1);
    free(path);
  }
  return ncclSuccess;
}
//在 NCCL 拓扑 XML 结构中查找或创建一个 net 节点（网络设备），并将其挂载到合适的父节点（如 PCI、CPU）下。
ncclResult_t ncclTopoFillNet(struct ncclXml* xml, const char* pciPath, const char* netName, struct ncclXmlNode** netNode, struct ncclXmlNode* forceParent) {
  NCCLCHECK(xmlFindTagKv(xml, "net", netNode, "name", netName));

  if (*netNode != NULL) return ncclSuccess;

  const char* pciSysPath = pciPath;
  //如果提供了 pciPath，判断其子系统类型（如 pci、usb、virtual、net 等）
  if (pciSysPath) {
    char subSystem[PATH_MAX];
    NCCLCHECK(ncclTopoGetSubsystem(pciSysPath, subSystem));
    // This is not a PCI device (virtual, usb, ...).
    /*
    一般来说，/sys/class/net/ethX/subsystem 指向的是 /sys/class/net，不是物理设备的subsystem
    但这里用的是“pci路径”的subsystem，也就是 /sys/devices/pciXXXX:XX/XXXX:XX:XX.X/subsystem，这个一定是“pci”或者其他物理总线类型（如usb、platform等），不会是“net”。
    这些PCI设备的subsystem通常就是“pci”，而不是“net”。“net”是网络接口的分类，而“pci”是物理设备的分类。
    如果不是（比如subSystem是“virtual”、“usb”等），那它就不是一个标准的PCI设备，不能参与PCI拓扑建模，这时就只能把它挂到CPU节点下，作为一个“特殊”设备处理。
    */
    if (strcmp(subSystem, "pci") != 0) {
      INFO(NCCL_NET|NCCL_GRAPH, "Topology detection: network path %s is not a PCI device (%s). Attaching to first CPU", pciSysPath, subSystem);
      pciSysPath = NULL;
    }
  }

  struct ncclXmlNode* parent = NULL;
  if (forceParent) {
    parent = forceParent;//如果传入了 forceParent，则直接用它作为父节点。
  } else if (pciSysPath) {//如果是 PCI 设备，则通过 busId 查找或创建对应的 PCI 节点作为父节点，并补全其属性。
    int offset;
    for (offset=strlen(pciSysPath)-1; pciSysPath[offset] != '/'; offset--);
    char busId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
    strcpy(busId, pciSysPath+offset+1);
    NCCLCHECK(ncclTopoGetPciNode(xml, busId, &parent));
    NCCLCHECK(xmlSetAttrIfUnset(parent, "class", "0x02"));
    NCCLCHECK(ncclTopoGetXmlFromSys(parent, xml));//根据 PCI 节点的 busId，从系统 /sys 路径自动补全该 PCI 设备的详细属性，并自动挂载到正确的父节点（如上级 PCI 交换机或 CPU） 。
  } else {
    // Virtual NIC, no PCI device, attach to first CPU。 如果不是 PCI 设备（如虚拟网卡），则挂载到第一个CPU 节点下。
    NCCLCHECK(xmlFindTag(xml, "cpu", &parent));
  }

//查找 nic 子节点，如果没有则新建。
  struct ncclXmlNode* nicNode = NULL;
  NCCLCHECK(xmlGetSub(parent, "nic", &nicNode));
  if (nicNode == NULL) {
    NCCLCHECK(xmlAddNode(xml, parent, "nic", &nicNode));
  }

  // We know that this net does not exist yet (we searched for it at the
  // beginning of this function), so we can add it.
  // 在 nic 节点下新建 net 节点，并设置其 name 属性。
  NCCLCHECK(xmlAddNode(xml, nicNode, "net", netNode));
  NCCLCHECK(xmlSetAttr(*netNode, "name", netName));
  return ncclSuccess;
}

ncclResult_t ncclTopoTrimXmlRec(struct ncclXmlNode* node, int* keep) {
  const char* str;
  NCCLCHECK(xmlGetAttr(node, "keep", &str));
  if (str && strcmp(str, "1") == 0) {
    NCCLCHECK(xmlUnsetAttr(node, "keep"));
    *keep = 1;
  } else {
    // Copy nSubs and subs as they could change as we trim recursively.
    struct ncclXmlNode* subs[MAX_SUBS];
    int nSubs = node->nSubs;
    memcpy(subs, node->subs, node->nSubs*sizeof(struct ncclXmlNode*));
    *keep = 0;
    for (int s=0; s<nSubs; s++) {
      int k = 0;
      NCCLCHECK(ncclTopoTrimXmlRec(subs[s], &k));
      *keep += k;
    }
    if (*keep == 0 && // Trim PCI switches or CPU with no used GPU/NIC under them.
        (strcmp(node->name, "pci") == 0 || strcmp(node->name, "cpu") == 0)) {
      NCCLCHECK(xmlRemoveNode(node));
    }
  }
  return ncclSuccess;
}
ncclResult_t ncclTopoTrimXml(struct ncclXml* xml) {
  int keep = 0;
  NCCLCHECK(ncclTopoTrimXmlRec(xml->nodes, &keep));
  return ncclSuccess;
}

/**************************************************/
/* Parser rules for the user-defined graph search */
/**************************************************/

ncclResult_t ncclTopoXmlGraphLoadGpu(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  NCCLCHECK(xmlLoadSub(file, xml, head, NULL, 0));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlGraphLoadNet(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  NCCLCHECK(xmlLoadSub(file, xml, head, NULL, 0));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlGraphLoadChannel(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  struct xmlHandler handlers[] = { { "net", ncclTopoXmlGraphLoadNet }, { "gpu", ncclTopoXmlGraphLoadGpu } };
  NCCLCHECK(xmlLoadSub(file, xml, head, handlers, 2));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlGraphLoadGraph(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  struct xmlHandler handlers[] = { { "channel", ncclTopoXmlGraphLoadChannel } };
  NCCLCHECK(xmlLoadSub(file, xml, head, handlers, 1));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlGraphLoadGraphs(FILE* file, struct ncclXml* xmlGraph, struct ncclXmlNode* head) {
  int version;
  NCCLCHECK(xmlGetAttrInt(head, "version", &version));
  if (version != NCCL_GRAPH_XML_VERSION) {
    WARN("XML Graph has wrong version %d, %d needed", version, NCCL_GRAPH_XML_VERSION);
    return ncclInvalidUsage;
  }
  const char* name;
  NCCLCHECK(xmlGetAttr(head, "name", &name));
  if (name != NULL) INFO(NCCL_GRAPH, "Loading graphs for topology %s", name);
  else INFO(NCCL_GRAPH, "Loading graphs");

  struct xmlHandler handlers[] = { { "graph", ncclTopoXmlGraphLoadGraph } };
  NCCLCHECK(xmlLoadSub(file, xmlGraph, head, handlers, 1));
  return ncclSuccess;
}

ncclResult_t ncclTopoGetXmlGraphFromFile(const char* xmlGraphFile, struct ncclXml* xml) {
  FILE* file = fopen(xmlGraphFile, "r");
  if (file == NULL) {
    WARN("Could not open XML graph file %s : %s", xmlGraphFile, strerror(errno));
    return ncclSystemError;
  }
  struct xmlHandler handlers[] = { { "graphs", ncclTopoXmlGraphLoadGraphs } };
  xml->maxIndex = 0;
  NCCLCHECK(xmlLoadSub(file, xml, NULL, handlers, 1));
  fclose(file);
  return ncclSuccess;
}
