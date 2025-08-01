/*************************************************************************
 * Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef XML_H_
#define XML_H_

#include "nccl.h"
#include "debug.h"
#include "checks.h"
#include "alloc.h"
#include <stdlib.h>
//解析和操作 XML 拓扑描述文件的基本数据结构，包括节点的名称、属性、父子关系等，便于后续对 XML 拓扑结构的遍历和操作。
// A few constraints to make the implementation easy
#define MAX_STR_LEN 255 //定义字符串最大长度为255
#define MAX_ATTR_COUNT 16 //定义每个 XML 节点最多有16个属性 。用于限制每个节点的属性数量
#define MAX_SUBS 128 //定义每个 XML 节点最多有128个子节点 。用于限制 XML 树的分支宽度。

#define NODE_TYPE_NONE 0 //无类型（未初始化或已删除节点）
#define NODE_TYPE_OPEN 1 //开放节点（如 <tag> ）
#define NODE_TYPE_CLOSE 2 //关闭节点（如 </tag> ）
#define NODE_TYPE_SINGLE 3 //自闭合节点（如 <tag/> ）
//xml 节点。 这里有个关键的信息：所有节点是在连续的一片地址空间，后续的跨进程融合xml非常有用。
struct ncclXmlNode {
  char name[MAX_STR_LEN+1];
  struct {
    char key[MAX_STR_LEN+1];
    char value[MAX_STR_LEN+1];
  } attrs[MAX_ATTR_COUNT+1]; // Need an extra one to consume extra params 额外的1个用于处理特殊情况（如多余参数）
  int nAttrs;//属性数量
  int type;//节点类型 。取值为上面定义的 NODE_TYPE_* 宏之一
  struct ncclXmlNode* parent;//父节点指针
  struct ncclXmlNode* subs[MAX_SUBS];//子节点指针数组 。每个元素指向一个子节点
  int nSubs;//子节点数量
};
//XML 文档结构体，文档容器，不是标签
struct ncclXml {
  int maxIndex, maxNodes;//maxIndex:当前已分配的节点数(下一个可用节点的索引)。maxNodes:最大可分配节点数（数组容量）
  struct ncclXmlNode nodes[1];//节点数组，实际分配时会根据需要扩展为更大的数组（C语言常用的“柔性数组”技巧）。
  //这里的nodes存储的是所有的节点，不止根节点。
};

/* File functions */
#define NCCL_TOPO_XML_VERSION 1
ncclResult_t ncclTopoGetXmlFromFile(const char* xmlTopoFile, struct ncclXml* xml, int warn);
ncclResult_t ncclTopoDumpXmlToFile(const char* xmlTopoFile, struct ncclXml* xml);
#define NCCL_GRAPH_XML_VERSION 1
ncclResult_t ncclTopoGetXmlGraphFromFile(const char* xmlGraphFile, struct ncclXml* xml);

/* Auto-detect functions */
//自动检测并填充 GPU 节点信息.busId:GPU 的总线 ID
ncclResult_t ncclTopoFillGpu(struct ncclXml* xml, const char* busId, struct ncclXmlNode** gpuNode);
//自动检测并填充网络（NIC）节点信息.pciPath:PCI 设备路径。netName:网络名称。forceParent:可选，指定父节点（如需强制挂载到某个父节点）。
ncclResult_t ncclTopoFillNet(struct ncclXml* xml, const char* pciPath, const char* netName, struct ncclXmlNode** netNode, struct ncclXmlNode* forceParent=NULL);

/* Remove unneeded parts  移除不需要的 XML 节点*/ 
ncclResult_t ncclTopoTrimXml(struct ncclXml* xml);

/* Fuse multiple system XMLs into one, skipping duplicate entries  合并多个系统的 XML 拓扑，跳过重复节点。将多个 XML 拓扑合并为一个，避免重复*/
ncclResult_t ncclTopoFuseXml(struct ncclXml* dst, struct ncclXml* src);
/* Relocate pointers in XML to (de-)serialize the structure 重定位 XML 内部指针，实现序列化/反序列化 将 XML 结构体中的指针转换为偏移量（或反之），便于存储和传输*/
ncclResult_t ncclTopoConvertXml(struct ncclXml* xml, uintptr_t base, int exp);

/**************/
/* XML Struct */
/* Functions  */
/**************/

static size_t xmlMemSize(int maxNodes) {
  return offsetof(struct ncclXml, nodes) + sizeof(struct ncclXmlNode)*maxNodes;
}
static ncclResult_t xmlAlloc(struct ncclXml** xml, int maxNodes) {
  char* mem;
  NCCLCHECK(ncclCalloc(&mem, xmlMemSize(maxNodes)));
  *xml = (struct ncclXml*)mem;
  (*xml)->maxNodes = maxNodes;
  return ncclSuccess;
}

static ncclResult_t xmlGetAttrIndex(struct ncclXmlNode* node, const char* attrName, int* index) {
  *index = -1;
  const int nAttrs = node->nAttrs;
  for (int a=0; a<nAttrs; a++) {
    if (strncmp(node->attrs[a].key, attrName, MAX_STR_LEN) == 0) {
      *index = a;
      return ncclSuccess;
    }
  }
  return ncclSuccess;
}

static ncclResult_t xmlGetAttr(struct ncclXmlNode* node, const char* attrName, const char** value) {
  int index;
  NCCLCHECK(xmlGetAttrIndex(node, attrName, &index));
  *value = index == -1 ? NULL : node->attrs[index].value;
  return ncclSuccess;
}

static ncclResult_t xmlGetAttrStr(struct ncclXmlNode* node, const char* attrName, const char** value) {
  NCCLCHECK(xmlGetAttr(node, attrName, value));
  if (*value == NULL) {
    WARN("Attribute %s of node %s not found", attrName, node->name);
    return ncclInternalError;
  }
  return ncclSuccess;
}
static ncclResult_t xmlGetAttrInt(struct ncclXmlNode* node, const char* attrName, int* value) {
  const char* str;
  NCCLCHECK(xmlGetAttrStr(node, attrName, &str));
  *value = strtol(str, NULL, 0);
  return ncclSuccess;
}

static ncclResult_t xmlGetAttrIntDefault(struct ncclXmlNode* node, const char* attrName, int* value, int defaultValue) {
  const char* str;
  NCCLCHECK(xmlGetAttr(node, attrName, &str));
  *value = str ? strtol(str, NULL, 0) : defaultValue;
  return ncclSuccess;
}

static ncclResult_t xmlGetAttrLong(struct ncclXmlNode* node, const char* attrName, int64_t* value) {
  const char* str;
  NCCLCHECK(xmlGetAttrStr(node, attrName, &str));
  *value = strtol(str, NULL, 0);
  return ncclSuccess;
}


static ncclResult_t xmlGetAttrFloat(struct ncclXmlNode* node, const char* attrName, float* value) {
  const char* str;
  NCCLCHECK(xmlGetAttrStr(node, attrName, &str));
  *value = strtof(str, NULL);
  return ncclSuccess;
}

static ncclResult_t xmlGetAttrFloatDefault(struct ncclXmlNode* node, const char* attrName, float* value, float defaultValue) {
  const char* str;
  NCCLCHECK(xmlGetAttr(node, attrName, &str));
  *value = str ? strtof(str, NULL) : defaultValue;
  return ncclSuccess;
}

static ncclResult_t xmlFindTag(struct ncclXml* xml, const char* tagName, struct ncclXmlNode** node) {
  *node = NULL;
  for (int i=0; i<xml->maxIndex; i++) {
    struct ncclXmlNode* n = xml->nodes+i;
    if (strcmp(n->name, tagName) == 0) {
      *node = n;
      return ncclSuccess;
    }
  }
  return ncclSuccess;
}

static ncclResult_t xmlFindNextTag(struct ncclXml* xml, const char* tagName, struct ncclXmlNode* prev, struct ncclXmlNode** node) {
  *node = NULL;
  for (int i=prev-xml->nodes+1; i<xml->maxIndex; i++) {
    struct ncclXmlNode* n = xml->nodes+i;
    if (strcmp(n->name, tagName) == 0) {
      *node = n;
      return ncclSuccess;
    }
  }
  return ncclSuccess;
}
//在 XML 节点数组中查找第一个满足指定标签名和属性键值对的节点 。
static ncclResult_t xmlFindTagKv(struct ncclXml* xml, const char* tagName, struct ncclXmlNode** node, const char* attrName, const char* attrValue) {
  *node = NULL;
  for (int i=0; i<xml->maxIndex; i++) {
    struct ncclXmlNode* n = xml->nodes+i;
    if (strcmp(n->name, tagName) == 0) {
      const char* value;
      NCCLCHECK(xmlGetAttr(n, attrName, &value));
      if (value && strcmp(value, attrValue) == 0) {
        *node = n;
        return ncclSuccess;
      }
    }
  }
  return ncclSuccess;
}

static ncclResult_t xmlFindNode(struct ncclXmlNode* parentNode, struct ncclXmlNode* searchNode, struct ncclXmlNode** node) {
  *node = NULL;
  // Search for the node at the current level only.
  for (int i=0; i<parentNode->nSubs; i++) {
    struct ncclXmlNode* n = parentNode->subs[i];
    if (strcmp(n->name, searchNode->name) == 0 && n->type == searchNode->type && n->nAttrs == searchNode->nAttrs) {
      int a;
      // Ensure that all the attributes are the same.
      for (a=0; a<searchNode->nAttrs; a++) {
        const char* val;
        NCCLCHECK(xmlGetAttr(n, searchNode->attrs[a].key, &val));
        if (!val || strcmp(val, searchNode->attrs[a].value))
          break;
      }
      if (a == searchNode->nAttrs) {
        *node = n;
        return ncclSuccess;
      }
    }
  }
  return ncclSuccess;
}

static ncclResult_t xmlSetAttr(struct ncclXmlNode* node, const char* attrName, const char* value) {
  int index;
  NCCLCHECK(xmlGetAttrIndex(node, attrName, &index));
  if (index == -1) {
    index = node->nAttrs++;
    strncpy(node->attrs[index].key, attrName, MAX_STR_LEN);
    node->attrs[index].key[MAX_STR_LEN] = '\0';
  }
  strncpy(node->attrs[index].value, value, MAX_STR_LEN);
  node->attrs[index].value[MAX_STR_LEN] = '\0';
  return ncclSuccess;
}

static ncclResult_t xmlPrintNodeRecursive(struct ncclXmlNode* node, const char* name) {
  while (node) {
    char line[1024*8];
    int cursor = 0;
    snprintf(line, sizeof(line), "<name=%s", node->name);
    for (int i = 0; i < node->nAttrs; i++) {
      cursor = strlen(line);
      snprintf(line + cursor, sizeof(line) - cursor, " %s=%s", node->attrs[i].key, node->attrs[i].value);
    }
    cursor = strlen(line);
    snprintf(line + cursor, sizeof(line) - cursor, ">");
    INFO(NCCL_GRAPH, "%s", line);
    node = node->parent;
  }
  return ncclSuccess;
}


static ncclResult_t xmlSetAttrIfUnset(struct ncclXmlNode* node, const char* attrName, const char* value) {
  int index;
  NCCLCHECK(xmlGetAttrIndex(node, attrName, &index));
  if (index != -1) return ncclSuccess;
  index = node->nAttrs++;
  strncpy(node->attrs[index].key, attrName, MAX_STR_LEN);
  node->attrs[index].key[MAX_STR_LEN] = '\0';
  strncpy(node->attrs[index].value, value, MAX_STR_LEN);
  node->attrs[index].value[MAX_STR_LEN] = '\0';
  return ncclSuccess;
}

static ncclResult_t xmlSetAttrInt(struct ncclXmlNode* node, const char* attrName, const int value) {
  int index;
  NCCLCHECK(xmlGetAttrIndex(node, attrName, &index));
  if (index == -1) {
    index = node->nAttrs++;
    strncpy(node->attrs[index].key, attrName, MAX_STR_LEN);
    node->attrs[index].key[MAX_STR_LEN] = '\0';//手动防止不合法的字符
  }
  snprintf(node->attrs[index].value, MAX_STR_LEN, "%d", value);
  node->attrs[index].value[MAX_STR_LEN] = '\0';
  return ncclSuccess;
}

static ncclResult_t xmlSetAttrFloat(struct ncclXmlNode* node, const char* attrName, const float value) {
  int index;
  NCCLCHECK(xmlGetAttrIndex(node, attrName, &index));
  if (index == -1) {
    index = node->nAttrs++;
    strncpy(node->attrs[index].key, attrName, MAX_STR_LEN);
    node->attrs[index].key[MAX_STR_LEN] = '\0';
  }
  snprintf(node->attrs[index].value, MAX_STR_LEN, "%g", value);
  node->attrs[index].value[MAX_STR_LEN] = '\0';
  return ncclSuccess;
}

static ncclResult_t xmlSetAttrLong(struct ncclXmlNode* node, const char* attrName, const int64_t value) {
  int index;
  NCCLCHECK(xmlGetAttrIndex(node, attrName, &index));
  if (index == -1) {
    index = node->nAttrs++;
    strncpy(node->attrs[index].key, attrName, MAX_STR_LEN);
    node->attrs[index].key[MAX_STR_LEN] = '\0';
  }
  snprintf(node->attrs[index].value, MAX_STR_LEN, "%#lx", value);
  node->attrs[index].value[MAX_STR_LEN] = '\0';
  return ncclSuccess;
}

static ncclResult_t xmlUnsetAttr(struct ncclXmlNode* node, const char* attrName) {
  int index;
  NCCLCHECK(xmlGetAttrIndex(node, attrName, &index));
  if (index == -1) return ncclSuccess;
  for (int i=index+1; i<node->nAttrs; i++) {
    strcpy(node->attrs[i-1].key, node->attrs[i].key);
    strcpy(node->attrs[i-1].value, node->attrs[i].value);
  }
  node->nAttrs--;
  return ncclSuccess;
}

static ncclResult_t xmlGetSub(struct ncclXmlNode* node, const char* subName, struct ncclXmlNode** sub) {
  *sub = NULL;
  for (int s=0; s<node->nSubs; s++) {
    if (strcmp(node->subs[s]->name, subName) == 0) {
      *sub = node->subs[s];
      return ncclSuccess;
    }
  }
  return ncclSuccess;
}
//从node中查找子节点，要求子节点满足名字，以及kv的要求。如果找到，就赋值给子节点sub，否则sub就是NULL。
static ncclResult_t xmlGetSubKv(struct ncclXmlNode* node, const char* subName, struct ncclXmlNode** sub, const char* attrName, const char* attrValue) {
  *sub = NULL;
  for (int s=0; s<node->nSubs; s++) {
    struct ncclXmlNode* subNode = node->subs[s];
    if (strcmp(subNode->name, subName) == 0) {
      const char* value;
      NCCLCHECK(xmlGetAttr(subNode, attrName, &value));
      if (value && strcmp(value, attrValue) == 0) {
        *sub = node->subs[s];
        return ncclSuccess;
      }
    }
  }
  return ncclSuccess;
}
static ncclResult_t xmlGetSubKvInt(struct ncclXmlNode* node, const char* subName, struct ncclXmlNode** sub, const char* attrName, const int attrValue) {
  char strValue[10];
  snprintf(strValue, 10, "%d", attrValue);
  NCCLCHECK(xmlGetSubKv(node, subName, sub, attrName, strValue));
  return ncclSuccess;
}
//添加一个节点，其实就是把创建好的节点的地址赋给sub
static ncclResult_t xmlAddNode(struct ncclXml* xml, struct ncclXmlNode* parent, const char* subName, struct ncclXmlNode** sub) {
  if (xml->maxIndex == xml->maxNodes) {
    WARN("Error : too many XML nodes (max %d)", xml->maxNodes);
    return ncclInternalError;
  }
  struct ncclXmlNode* s = xml->nodes+xml->maxIndex++;
  s->nSubs = 0;
  s->nAttrs = 0;
  *sub = s;
  s->parent = parent;
  if (parent) {
    if (parent->nSubs == MAX_SUBS) {
      WARN("Error : too many XML subnodes (max %d)", MAX_SUBS);
      return ncclInternalError;
    }
    parent->subs[parent->nSubs++] = s;
  }
  strncpy(s->name, subName, MAX_STR_LEN);//长度小于n的时候用'\0'填充
  s->name[MAX_STR_LEN] = '\0';
  return ncclSuccess;
}

static ncclResult_t xmlRemoveNode(struct ncclXmlNode* node) {
  node->type = NODE_TYPE_NONE;
  struct ncclXmlNode* parent = node->parent;
  if (parent == NULL) return ncclSuccess;
  int shift = 0;
  for (int s=0; s<parent->nSubs; s++) {
    if (parent->subs[s] == node) shift = 1;
    else if (shift) parent->subs[s-1] = parent->subs[s];
  }
  parent->nSubs--;
  return ncclSuccess;
}
// 它的作用是递归地将一个 XML 子树（ srcNode ）拷贝到目标 XML 树（ dst ）的指定父节点（ parent ）下
static ncclResult_t xmlAddTree(struct ncclXml* dst, struct ncclXmlNode* parent, struct ncclXmlNode* srcNode) {
  if (dst->maxIndex == dst->maxNodes) {//检查目标 XML 节点数组是否已满，防止越界。如果满了就报错并返回。
    WARN("Error : too many XML nodes (max %d)", dst->maxNodes);
    return ncclInternalError;
  }
  //在目标 XML 的节点数组中分配一个新节点，并将 dstNode 指向它。
  struct ncclXmlNode* dstNode = dst->nodes+dst->maxIndex++;
  // 直接拷贝源节点的内容到新分配的目标节点（浅拷贝，包括属性、名字等）。
  // 注意这里是拷贝！！！！因为dst中的Node空间早就分配好了。
  *dstNode = *srcNode;//
  dstNode->parent = parent;//设置新节点的父节点指针。
  //如果有父节点，把新节点挂到父节点的子节点数组里，并增加父节点的子节点数量。
  if (parent) {
    if (parent->nSubs == MAX_SUBS) {
      WARN("Error : too many XML subnodes (max %d)", MAX_SUBS);
      return ncclInternalError;
    }
    parent->subs[parent->nSubs++] = dstNode;
  }
  // 这里把新节点的子节点数量清零，准备递归添加子树。
  dstNode->nSubs = 0;
  // Recursively copy the subtree(s)
  for (int i=0; i<srcNode->nSubs; i++)
    NCCLCHECK(xmlAddTree(dst, dstNode, srcNode->subs[i]));
  return ncclSuccess;
}


// Dictionary for STR -> INT conversions. No dictionary size information,
// there needs to be a last element with str == NULL.
struct kvDict {
  const char* str;
  int value;
};

static ncclResult_t kvConvertToInt(const char* str, int* value, struct kvDict* dict) {
  struct kvDict* d = dict;
  while (d->str) {
    if (strncmp(str, d->str, strlen(d->str)) == 0) {
      *value = d->value;
      return ncclSuccess;
    }
    d++;
  }
  INFO(NCCL_GRAPH, "KV Convert to int : could not find value of '%s' in dictionary, falling back to %d", str, d->value);
  *value = d->value;
  return ncclSuccess;
}
static ncclResult_t kvConvertToStr(int value, const char** str, struct kvDict* dict) {
  struct kvDict* d = dict;
  while (d->str) {
    if (value == d->value) {
      *str = d->str;
      return ncclSuccess;
    }
    d++;
  }
  WARN("KV Convert to str : could not find value %d in dictionary", value);
  return ncclInternalError;
}

#endif
