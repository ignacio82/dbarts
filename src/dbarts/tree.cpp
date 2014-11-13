#include "config.hpp"
#include "tree.hpp"

#include <cerrno>
#include <cstdlib> // strtol
// #include <cstring> //
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

#include <external/alloca.h>
#include <external/binaryIO.h>
#include <external/stats.h>
#include <external/stringWriter.h>

#include <dbarts/bartFit.hpp>
#include <dbarts/data.hpp>
#include <dbarts/endNodeModel.hpp>
#include <dbarts/scratch.hpp>
#include <dbarts/state.hpp>
#include "functions.hpp"

using std::uint32_t;

namespace {
  using namespace dbarts;
  
  // multithread me!
  size_t* createObservationToNodeIndexMap(const BARTFit& fit, const Node& top,
                                          const double* Xt, size_t numObservations)
  {
    if (numObservations == 0) return NULL;
    
    size_t* map = new size_t[numObservations];
        
    for (size_t i = 0; i < numObservations; ++i) {
      const Node* bottomNode = top.findBottomNode(fit, Xt + i * fit.data.numPredictors);
      
      map[i] = bottomNode->enumerationIndex;
    }
    
    return map;
  }
}

namespace dbarts {
  // void Tree::drawFromEndNodePosteriorsAndSetFits(const BARTFit& fit, const double* y, double residualVariance)
  // void Tree::
  
  void Tree::sampleAveragesAndSetFits(const BARTFit& fit, const double* y, double* trainingFits, double* testFits)
  {
    NodeVector bottomNodes(getAndEnumerateBottomVector());
    size_t numBottomNodes = bottomNodes.size();
    
    double* nodePosteriorPredictions = NULL;
    
    if (testFits != NULL) nodePosteriorPredictions = ext_stackAllocate(numBottomNodes, double);
    
    for (size_t i = 0; i < numBottomNodes; ++i) {
      const Node& bottomNode(*bottomNodes[i]);
      
      bottomNode.drawFromPosterior(fit, y, fit.state.sigma * fit.state.sigma);
      bottomNode.getPredictions(fit, NULL, NULL, trainingFits);
      
      if (testFits != NULL) nodePosteriorPredictions[i] = trainingFits[bottomNode.isTop() ? 0 : bottomNode.observationIndices[0]];
    }
    
    if (testFits != NULL) {
      size_t* observationNodeMap = createObservationToNodeIndexMap(fit, *this, fit.scratch.Xt_test, fit.data.numTestObservations);
      for (size_t i = 0; i < fit.data.numTestObservations; ++i) testFits[i] = nodePosteriorPredictions[observationNodeMap[i]];
      delete [] observationNodeMap;
      
      ext_stackFree(nodePosteriorPredictions);
    }
  }
  
  double* Tree::recoverAveragesFromFits(const BARTFit&, const double* treeFits)
  {
    NodeVector bottomNodes(getBottomVector());
    size_t numBottomNodes = bottomNodes.size();
    
    double* result = new double[numBottomNodes];
    for (size_t i = 0; i < numBottomNodes; ++i) {
      if (bottomNodes[i]->isTop()) {
        result[i] = treeFits[0];
      } else if (bottomNodes[i]->getNumObservations() > 0) {
        result[i] = treeFits[bottomNodes[i]->observationIndices[0]];
      } else {
        result[i] = 0.0;
      }
    }
    
    return(result);
  }
  
  void Tree::setCurrentFitsFromAverages(const BARTFit& fit, const double* posteriorPredictions, double* trainingFits, double* testFits)
  {
    NodeVector bottomNodes(getAndEnumerateBottomVector());
    size_t numBottomNodes = bottomNodes.size();
    
    if (trainingFits != NULL) {
      for (size_t i = 0; i < numBottomNodes; ++i) {
        const Node& bottomNode(*bottomNodes[i]);
        
        // bottomNode.setPosterior
        *static_cast<double*>(bottomNode.getScratch()) = posteriorPredictions[i];
        bottomNode.getPredictions(fit, NULL, NULL, trainingFits);
      }
    }
    
    if (testFits != NULL) {
      size_t* observationNodeMap = createObservationToNodeIndexMap(fit, *this, fit.scratch.Xt_test, fit.data.numTestObservations);
      for (size_t i = 0; i < fit.data.numTestObservations; ++i) testFits[i] = posteriorPredictions[observationNodeMap[i]];
      delete [] observationNodeMap;
    }
  }
  
  bool Tree::isValid() const {
    const NodeVector bottomNodes(getBottomVector());
    size_t numBottomNodes = bottomNodes.size();
    
    for (size_t j = 0; j < numBottomNodes; ++j) {
      if (bottomNodes[j]->getNumObservations() == 0) return false;
    }
    
    return true;
  }
}

namespace {
  using namespace dbarts;
  
  int writeNode(ext_stringWriter* writer, const Node& node) {
    if (node.isBottom()) return ext_swr_writeChar(writer, '.');
    
    int errorCode = 0;
    if ((errorCode = ext_swr_write32BitInteger(writer, node.rule.variableIndex)) != 0) return errorCode;
    if ((errorCode = ext_swr_writeChar(writer, ' ')) != 0) return errorCode;
    if ((errorCode = ext_swr_write32BitInteger(writer, node.rule.splitIndex)) != 0) return errorCode;
    if ((errorCode = ext_swr_writeChar(writer, ' ')) != 0) return errorCode;
    
    if ((errorCode = writeNode(writer, *node.getLeftChild())) != 0) return errorCode;
    errorCode = writeNode(writer, *node.getRightChild());
    
    return errorCode;
  }
  
  int readNode(const BARTFit& fit, Node&node, const char* string, size_t* bytesRead)
  {
    if (string[0] == '\0') return (*bytesRead = 0, 0);
    if (string[0] == '.') return (*bytesRead = 1, 0);
    
    size_t pos = 0;
    
#define INT_BUFFER_LENGTH 16
    char buffer[INT_BUFFER_LENGTH];
    while (string[pos] != ' ' && pos < INT_BUFFER_LENGTH) {
      buffer[pos] = string[pos];
      ++pos;
    }
    
    if (pos == INT_BUFFER_LENGTH) return (*bytesRead = pos, ENOBUFS);
    buffer[pos++] = '\0';
    
    
    errno = 0;
    node.rule.variableIndex = static_cast<int32_t>(std::strtol(buffer, NULL, 10));
    if (node.rule.variableIndex == 0 && errno != 0) return errno;
    
    // ext_throwError("unable to parse tree string: expected integer");
    // ext_throwError("unable to parse tree string: %s", strerror(errno));
    
    size_t bufferPos = 0;
    while (string[pos] != ' ' && bufferPos < INT_BUFFER_LENGTH) {
      buffer[bufferPos++] = string[pos++];
    }
    
    if (pos == INT_BUFFER_LENGTH) return (*bytesRead = pos, ENOBUFS);
    buffer[bufferPos++] = '\0';
    ++pos;
    
    errno = 0;
    node.rule.splitIndex = static_cast<int32_t>(strtol(buffer, NULL, 10));
    if (node.rule.splitIndex == 0 && errno != 0) return (*bytesRead = pos, ENOBUFS);
    
    node.leftChild    = Node::create(fit, node);
    node.rightChild = Node::create(fit, node);
    
    size_t childBytesRead;
    
    int errorCode = readNode(fit, *node.getLeftChild(), string + pos, &childBytesRead);
    pos += childBytesRead;
    
    if (errorCode != 0) return (*bytesRead = pos, errorCode);
    
    errorCode = readNode(fit, *node.getRightChild(), string + pos, &childBytesRead);
    pos += childBytesRead;
    
    *bytesRead = pos;
    return errorCode;
  }
  
#define NODE_HAS_CHILDREN 1
  int writeNode(const BARTFit& fit, const Node& node, ext_binaryIO* bio, const size_t* treeIndices) {
    int errorCode = 0;
    
    const Data& data(fit.data);
    
    ptrdiff_t observationOffset = 0;
    unsigned char nodeFlags = 0;
    uint64_t variablesAvailableForSplit = 0;
    
    observationOffset = node.observationIndices - treeIndices;
    if (observationOffset < 0) {
      errorCode = EINVAL; goto write_node_cleanup;
    } else if ((errorCode = ext_bio_writeSizeType(bio, static_cast<size_t>(observationOffset))) != 0) goto write_node_cleanup;
    
    if ((errorCode = ext_bio_writeSizeType(bio, node.enumerationIndex)) != 0) goto write_node_cleanup;
    if ((errorCode = ext_bio_writeSizeType(bio, node.numObservations)) != 0) goto write_node_cleanup;
    
    for (size_t i = 0; i < data.numPredictors; ++i) {
      if (node.variablesAvailableForSplit[i] == true) variablesAvailableForSplit |= 1 << i;
    }
    if ((errorCode = ext_bio_writeUnsigned64BitInteger(bio, variablesAvailableForSplit)) != 0) goto write_node_cleanup;
    
    if (node.leftChild != NULL) {
      nodeFlags += NODE_HAS_CHILDREN;
      
      if ((errorCode = ext_bio_writeChar(bio, *reinterpret_cast<char*>(&nodeFlags))) != 0) goto write_node_cleanup;
      
      if ((errorCode = ext_bio_writeUnsigned32BitInteger(bio, *reinterpret_cast<const uint32_t*>(&node.rule.variableIndex))) != 0) goto write_node_cleanup;
      if ((errorCode = ext_bio_writeUnsigned32BitInteger(bio, node.rule.categoryDirections)) != 0) goto write_node_cleanup;
      
      if ((errorCode = writeNode(fit, *node.leftChild, bio, treeIndices))) goto write_node_cleanup;
      if ((errorCode = writeNode(fit, *node.rightChild, bio, treeIndices))) goto write_node_cleanup;
    } else {
      if ((errorCode = ext_bio_writeChar(bio, *reinterpret_cast<char*>(&nodeFlags))) != 0) goto write_node_cleanup;
      
      if ((errorCode = fit.model.endNodeModel->writeScratch(node, bio)) != 0) goto write_node_cleanup;
    }
    
write_node_cleanup:
    
    return errorCode;
  }
  
  int readNode(const BARTFit& fit, Node& node, ext_binaryIO* bio, const size_t* treeIndices)
  {
    int errorCode = 0;
    
    const Data& data(fit.data);
    
    size_t observationOffset = 0;
    unsigned char nodeFlags = 0;
    uint64_t variablesAvailableForSplit = 0;
    dbarts::Node* leftChild = NULL;
    dbarts::Node* rightChild = NULL;
    
    if ((errorCode = ext_bio_readSizeType(bio, &observationOffset)) != 0) goto read_node_cleanup;
    if (observationOffset >= data.numObservations) { errorCode = EINVAL; goto read_node_cleanup; }
    node.observationIndices = const_cast<size_t*>(treeIndices) + observationOffset;
    
    if ((errorCode = ext_bio_readSizeType(bio, &node.enumerationIndex)) != 0) goto read_node_cleanup;
    if ((errorCode = ext_bio_readSizeType(bio, &node.numObservations)) != 0) goto read_node_cleanup;
    
    if ((errorCode = ext_bio_readUnsigned64BitInteger(bio, &variablesAvailableForSplit)) != 0) goto read_node_cleanup;
    for (size_t i = 0; i < data.numPredictors; ++i) {
      node.variablesAvailableForSplit[i] = (variablesAvailableForSplit & (1 << i)) != 0;
    }
    
    if ((errorCode = ext_bio_readChar(bio, reinterpret_cast<char*>(&nodeFlags))) != 0) goto read_node_cleanup;
    
    if (nodeFlags > NODE_HAS_CHILDREN) { errorCode = EINVAL; goto read_node_cleanup; }
    
    if (nodeFlags & NODE_HAS_CHILDREN) {
      if ((errorCode = ext_bio_readUnsigned32BitInteger(bio, reinterpret_cast<uint32_t*>(&node.rule.variableIndex))) != 0) goto read_node_cleanup;
      if ((errorCode = ext_bio_readUnsigned32BitInteger(bio, &node.rule.categoryDirections)) != 0) goto read_node_cleanup;
      
      leftChild = dbarts::Node::create(fit, node);
      node.leftChild = leftChild;
      if ((errorCode = readNode(fit, *leftChild, bio, treeIndices)) != 0) goto read_node_cleanup;
      
      rightChild = dbarts::Node::create(fit, node);
      node.rightChild = rightChild;
      if ((errorCode = readNode(fit, *rightChild, bio, treeIndices)) != 0) goto read_node_cleanup;
    } else {
      node.leftChild = NULL;
      
      if ((errorCode = fit.model.endNodeModel->readScratch(node, bio)) != 0) goto read_node_cleanup;
    }
    
read_node_cleanup:

    if (errorCode != 0) {
      delete rightChild;
      delete leftChild;
      
      node.leftChild = NULL;
    }
    
    return errorCode;
  }
}

namespace dbarts {
  int Tree::read(const BARTFit& fit, const char* string)
  {
    clear(fit);
    
    size_t pos = 0;
    int errorCode = readNode(fit, *this, string, &pos);
    
    if (errorCode != 0) return errorCode;
    
    if (!isBottom()) {
      updateVariablesAvailable(fit, *this, rule.variableIndex);
    }
    updateState(fit, NULL, BART_NODE_UPDATE_COVARIATES_CHANGED);
    
    return 0;
  }
  
  int Tree::read(const BARTFit& fit, ext_binaryIO* bio)
  {
    clear(fit);
    
    int errorCode = readNode(fit, *this, bio, getObservationIndices());
    
    if (errorCode != 0) return errorCode;
    
    if (!isBottom()) {
      updateVariablesAvailable(fit, *this, rule.variableIndex);
    }
    updateState(fit, NULL, BART_NODE_UPDATE_COVARIATES_CHANGED);
    
    return 0;
  }
  
  int Tree::write(const BARTFit& fit, ext_binaryIO* bio) const
  {
    return writeNode(fit, *this, bio, getObservationIndices());
  }
  
  int Tree::write(ext_stringWriter* writer) const
  {
    return writeNode(writer, *this);
  }
}
