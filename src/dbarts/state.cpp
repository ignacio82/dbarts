#include "config.hpp"
#include <dbarts/state.hpp>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#ifdef HAVE_SNPRINTF
#  include <cstdio>
#else
extern "C" {
#  include <stdio.h>
}
#endif

#ifdef HAVE_STD_SNPRINTF
using std::snprintf;
#endif

#include <external/io.h>

#include <dbarts/bartFit.hpp>
#include <dbarts/control.hpp>
#include "functions.hpp"
#include "node.hpp"
#include "tree.hpp"

#define BASE_BUFFER_SIZE 1024
#define INT_BUFFER_SIZE 16

namespace {
  using namespace dbarts;
  
  struct StringWriter {
    char* buffer;
    size_t length;
    size_t pos;
    
    void writeChar(char c) {
      buffer[pos++] = c;
      
      if (pos >= length) reallocate();
    }
    void writeString(const char* s, size_t len) {
      if (pos + len >= length) reallocate();
      std::memcpy(buffer + pos, s, len * sizeof(char));
      pos += len;
    }
    
    void writeInt(int32_t i) {
      char intBuffer[INT_BUFFER_SIZE];
      int bytesWritten = snprintf(intBuffer, INT_BUFFER_SIZE, "%d", i);
      writeString(intBuffer, (size_t) bytesWritten);
    }
    
    /* void writeUInt(uint32_t u) {
      char intBuffer[INT_BUFFER_SIZE];
      int bytesWritten = snprintf(intBuffer, INT_BUFFER_SIZE, "%u", u);
      writeString(intBuffer, (size_t) bytesWritten);
    } */
    
    void writeNode(const Node& node) {
      if (node.isBottom()) {
        writeChar('.');
        return;
      }
      
      writeInt(node.p.rule.variableIndex);
      writeChar(' ');
      writeInt(node.p.rule.splitIndex);
      writeChar(' ');
      
      writeNode(*node.getLeftChild());
      writeNode(*node.getRightChild());
    }
    
    void reallocate() {
      char* temp = new char[length + BASE_BUFFER_SIZE];
      std::memcpy(temp, (const char*) buffer, length * sizeof(char));
      
      length += BASE_BUFFER_SIZE;
      delete [] buffer;
      buffer = temp;
    }
  };
}

namespace {
  using namespace dbarts;
  
  size_t readNode(const BARTFit& fit, Node& node, const char* treeString, size_t numPredictors) {
    if (treeString[0] == '\0') return 0;
    if (treeString[0] == '.') return 1;
    
    
    size_t pos = 0;
    
    char buffer[INT_BUFFER_SIZE];
    while (treeString[pos] != ' ' && pos < INT_BUFFER_SIZE) {
      buffer[pos] = treeString[pos];
      ++pos;
    }
    
    if (pos == INT_BUFFER_SIZE) ext_throwError("Unable to parse tree string: expected integer.");
    buffer[pos++] = '\0';
    
    
    errno = 0;
    node.p.rule.variableIndex = (int32_t) strtol(buffer, NULL, 10);
    if (node.p.rule.variableIndex == 0 && errno != 0)
      ext_throwError("Unable to parse tree string: %s", strerror(errno));
    
    size_t bufferPos = 0;
    while (treeString[pos] != ' ' && bufferPos < INT_BUFFER_SIZE) {
      buffer[bufferPos++] = treeString[pos++];
    }
    
    if (pos == INT_BUFFER_SIZE) ext_throwError("Unable to parse tree string: expected integer.");
    buffer[bufferPos++] = '\0';
    ++pos;
    
    errno = 0;
    node.p.rule.splitIndex = (int32_t) strtol(buffer, NULL, 10);
    if (node.p.rule.splitIndex == 0 && errno != 0)
      ext_throwError("Unable to parse tree string: %s", strerror(errno));
    
    node.leftChild  = createNode(fit, node);
    node.p.rightChild = createNode(fit, node);
    
    pos += readNode(fit, *node.getLeftChild(), treeString + pos, numPredictors);
    pos += readNode(fit, *node.getRightChild(), treeString + pos, numPredictors);
    
    return pos;
  }
}


namespace dbarts {
  const char* const* State::createTreeStrings(const BARTFit& fit) const
  {
    StringWriter writer;
    
    char** result = new char*[fit.control.numTrees];
    for (size_t i = 0; i < fit.control.numTrees; ++i) {
      writer.buffer = new char[BASE_BUFFER_SIZE];
      writer.length = BASE_BUFFER_SIZE;
      writer.pos = 0;
      
      writer.writeNode(*NODE_AT(trees, i, fit.scratch.nodeSize));
     
      writer.writeChar('\0');
      
      result[i] = writer.buffer;
    }
    
    return(result);
  }
  
  void State::recreateTreesFromStrings(const BARTFit& fit, const char* const* treeStrings)
  {
    for (size_t i = 0; i < fit.control.numTrees; ++i) {
      NODE_AT(trees, i, fit.scratch.nodeSize)->clear(fit);
      readNode(fit, *NODE_AT(trees, i, fit.scratch.nodeSize), treeStrings[i], fit.data.numPredictors);
      updateVariablesAvailable(fit, *TREE_AT(trees, i, fit.scratch.nodeSize), TREE_AT(trees, i, fit.scratch.nodeSize)->p.rule.variableIndex);
      
      NODE_AT(trees, i, fit.scratch.nodeSize)->addObservationsToChildrenAndClearScratches(fit);
    }
  }
}
