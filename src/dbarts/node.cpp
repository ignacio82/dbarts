#include "config.hpp"
#include "node.hpp"

#include <cstring>    // memcpy
#include <algorithm>  // int max

#include <external/alloca.h>
#include <external/io.h>
#include <external/linearAlgebra.h>
#include <external/stats.h>
#include <external/stats_mt.h>

#include <dbarts/bartFit.hpp>
#include <dbarts/data.hpp>
#include <dbarts/endNodeModel.hpp>
#include <dbarts/scratch.hpp>
#include "functions.hpp"

using std::uint64_t;

namespace dbarts {
  
  double Rule::getSplitValue(const BARTFit& fit) const
  {
    if (variableIndex < 0) return -1000.0;
    if (fit.data.variableTypes[variableIndex] != ORDINAL) return -2000.0;
    
    return fit.scratch.cutPoints[variableIndex][splitIndex];
  }
  
  void Rule::invalidate() {
    variableIndex = DBARTS_INVALID_RULE_VARIABLE;
    splitIndex = DBARTS_INVALID_RULE_VARIABLE;
  }
  
  bool Rule::goesRight(const BARTFit& fit, const double* x) const
  {
    if (fit.data.variableTypes[variableIndex] == CATEGORICAL) {
      // x is a double, but that is 64 bits wide, and as such we can treat it as
      // a 64 bit integer
      uint32_t categoryId = static_cast<uint32_t>(*(reinterpret_cast<const uint64_t*>(x + variableIndex)));
      
      return categoryGoesRight(categoryId);
    } else {
      const double* splitValues = fit.scratch.cutPoints[variableIndex];
      
      return x[variableIndex] > splitValues[splitIndex];
    }
  }
  
  void Rule::copyFrom(const Rule& other)
  {
    if (other.variableIndex == DBARTS_INVALID_RULE_VARIABLE) {
      variableIndex = DBARTS_INVALID_RULE_VARIABLE;
      splitIndex    = DBARTS_INVALID_RULE_VARIABLE;
      return;
    }
    
    variableIndex = other.variableIndex;
    splitIndex    = other.splitIndex;
  }
  
  void Rule::swapWith(Rule& other)
  {
    Rule temp(other);
    other = *this;
    *this = temp;
  }

  bool Rule::equals(const Rule& other) const {
    if (variableIndex != other.variableIndex) return false;
    
    // since is a union of variables of the same width, bit-wise equality is sufficient
    return splitIndex == other.splitIndex;
  }
}
namespace {
  using namespace dbarts;
  
  void clearObservationsInNode(const BARTFit& fit, Node& node) {
    if (!node.isTop()) {
      node.observationIndices = NULL;
      node.numObservations = 0;
    }
    if (!node.isBottom()) {
      clearObservationsInNode(fit, *node.leftChild);
      clearObservationsInNode(fit, *node.p.rightChild);
    }
  }
}

namespace dbarts {
  void Node::clear(const BARTFit& fit)
  {
    if (!isBottom()) {
      Node::destroy(fit, leftChild);
      Node::destroy(fit, p.rightChild);
      
      leftChild = NULL;
      p.rule.invalidate();
    } else {
      fit.model.endNodeModel->deleteScratch(*this);
    }
    clearObservationsInNode(fit, *this);
  }
  
  /* Node* Node::create(const BARTFit& fit, size_t* observationIndices, size_t numObservations)
  {
    Node* result = static_cast<Node*>(::operator new(fit.scratch.nodeSize));

    Node::initialize(fit, *result, observationIndices, numObservations);

    return result;
  } */

  Node* Node::create(const BARTFit& fit, const Node& parent)
  {
    Node* result = static_cast<Node*>(::operator new(fit.scratch.nodeSize));
    result->parent = const_cast<Node*>(&parent);
    result->leftChild = NULL;
    result->enumerationIndex = BART_INVALID_NODE_ENUM;

    result->observationIndices = NULL;
    result->numObservations = 0;

    result->variablesAvailableForSplit = new bool[fit.data.numPredictors];
    std::memcpy(result->variablesAvailableForSplit, parent.variablesAvailableForSplit, sizeof(bool) * fit.data.numPredictors);
    
    fit.model.endNodeModel->createScratch(fit, *result);
    
    return result;
  }

  void Node::initialize(const BARTFit& fit, Node& node, size_t* observationIndices, size_t numObservations)
  {
    node.parent = NULL;
    node.leftChild = NULL;
    node.enumerationIndex = BART_INVALID_NODE_ENUM;

    node.observationIndices = observationIndices;
    node.numObservations = numObservations;

    node.variablesAvailableForSplit = new bool[fit.data.numPredictors];
    for (size_t i = 0; i < fit.data.numPredictors; ++i) node.variablesAvailableForSplit[i] = true;
    
    fit.model.endNodeModel->createScratch(fit, node);
  }

  void Node::destroy(const BARTFit& fit, Node* node)
  {
    Node::invalidate(fit, *node);

    ::operator delete (node);
  }
  
  void Node::invalidate(const BARTFit& fit, Node& node)
  {
    if (!node.isBottom()) {
      Node::destroy(fit, node.leftChild);
      Node::destroy(fit, node.p.rightChild);
    } else {
      fit.model.endNodeModel->deleteScratch(node);
    }
    delete [] node.variablesAvailableForSplit; node.variablesAvailableForSplit = NULL;
  }
  
  void Node::copyFrom(const BARTFit& fit, const Node& other)
  {
    parent = other.parent;
    leftChild = other.leftChild;
    
    enumerationIndex = other.enumerationIndex;
    std::memcpy(variablesAvailableForSplit, other.variablesAvailableForSplit, sizeof(bool) * fit.data.numPredictors);
    
    observationIndices = other.observationIndices;
    numObservations = other.numObservations;
    
    if (!isBottom()) {
      p.rightChild = other.p.rightChild;
      p.rule.copyFrom(other.p.rule);
    } else {
      fit.model.endNodeModel->copyScratch(fit, *this, other);
    }
  }
  
  void Node::print(const BARTFit& fit) const
  {
    size_t depth = getDepth();
        
    for (size_t i = 0; i < depth; ++i) ext_printf("  ");
    
    ext_printf("node:");
    ext_printf(" n: %lu", getNumObservations());
    ext_printf(" TBN: %u%u%u", isTop(), isBottom(), childrenAreBottom());
    ext_printf(" Avail: ");
    
    for (size_t i = 0; i < fit.data.numPredictors; ++i) ext_printf("%u", variablesAvailableForSplit[i]);
    
    if (!isBottom()) {
      ext_printf(" var: %d ", p.rule.variableIndex);
      
      if (fit.data.variableTypes[p.rule.variableIndex] == CATEGORICAL) {
        ext_printf("CATRule: ");
        for (size_t i = 0; 0 < fit.scratch.numCutsPerVariable[p.rule.variableIndex]; ++i) ext_printf(" %u", (p.rule.categoryDirections >> i) & 1);
      } else {
        ext_printf("ORDRule: (%d)=%f", p.rule.splitIndex, p.rule.getSplitValue(fit));
      }
    } else {
      fit.model.endNodeModel->printScratch(fit, *this);
    }
    ext_printf("\n");
    
    if (!isBottom()) {
      leftChild->print(fit);
      p.rightChild->print(fit);
    }
  }
  
  size_t Node::getNumBottomNodes() const
  {
    if (isBottom()) {
      return 1;
    } else {
      return leftChild->getNumBottomNodes() + p.rightChild->getNumBottomNodes();
    }
  }
  
  size_t Node::getNumNotBottomNodes() const
  {
    if (isBottom()) return 0;
    
    return leftChild->getNumNotBottomNodes() + p.rightChild->getNumNotBottomNodes() + 1;
  }
  
  size_t Node::getNumNoGrandNodes() const
  {
    if (isBottom()) return 0;
    if (childrenAreBottom()) return 1;
    return (leftChild->getNumNoGrandNodes() + p.rightChild->getNumNoGrandNodes());
  }
  
  size_t Node::getNumSwappableNodes() const
  {
    if (isBottom() || childrenAreBottom()) return 0;
    if ((leftChild->isBottom()  || leftChild->childrenAreBottom()) &&
        (p.rightChild->isBottom() || p.rightChild->childrenAreBottom())) return 1;
    
    return (leftChild->getNumSwappableNodes() + p.rightChild->getNumSwappableNodes() + 1);
  }
}

// NOTE: below assumes that walk tree on left
namespace {
  using namespace dbarts;
  
  void fillBottomVector(const Node& node, NodeVector& result)
  {
    if (node.isBottom()) {
      result.push_back(const_cast<Node*>(&node));
      return;
    }
    
    fillBottomVector(*node.leftChild, result);
    fillBottomVector(*node.p.rightChild, result);
  }
  
  void fillAndEnumerateBottomVector(Node& node, NodeVector& result, size_t& index)
  {
    if (node.isBottom()) {
      result.push_back(&node);
      node.enumerationIndex = index++;
      return;
    }
    
    fillAndEnumerateBottomVector(*node.getLeftChild(), result, index);
    fillAndEnumerateBottomVector(*node.getRightChild(), result, index);
  }
  
  void fillNoGrandVector(const Node& node, NodeVector& result)
  {
    if (node.isBottom()) return;
    if (node.childrenAreBottom()) {
      result.push_back(const_cast<Node*>(&node));
      return;
    }

    fillNoGrandVector(*node.getLeftChild(), result);
    fillNoGrandVector(*node.getRightChild(), result);
  }
  
  void fillNotBottomVector(const Node& node, NodeVector& result)
  {
    if (node.isBottom()) return;
    if (node.childrenAreBottom()) {
      result.push_back(const_cast<Node*>(&node));
      return;
    }
  
    fillNotBottomVector(*node.leftChild, result);
    fillNotBottomVector(*node.p.rightChild, result);
    
    result.push_back(const_cast<Node*>(&node));
  }
  
  void fillSwappableVector(const Node& node, NodeVector& result)
  {
    if (node.isBottom() || node.childrenAreBottom()) return;
    if ((node.leftChild->isBottom()  || node.leftChild->childrenAreBottom()) && 
        (node.p.rightChild->isBottom() || node.p.rightChild->childrenAreBottom())) {
      result.push_back(const_cast<Node*>(&node));
      return;
    }
    
    fillSwappableVector(*node.leftChild, result);
    fillSwappableVector(*node.p.rightChild, result);
    
    result.push_back(const_cast<Node*>(&node));
  }
}
namespace dbarts {
  NodeVector Node::getBottomVector() const
  {
    NodeVector result;
    fillBottomVector(*this, result);
    return result;
  }
  
  NodeVector Node::getAndEnumerateBottomVector()
  {
    size_t index = 0;
    NodeVector result;
    fillAndEnumerateBottomVector(*this, result, index);
    return result;
  }
  
  NodeVector Node::getNoGrandVector() const
  {
    NodeVector result;
    fillNoGrandVector(*this, result);
    return result;
  }
  
  NodeVector Node::getNotBottomVector() const
  {
    NodeVector result;
    fillNotBottomVector(*this, result);
    return result;
  }
  
  NodeVector Node::getSwappableVector() const
  {
    NodeVector result;
    fillSwappableVector(*this, result);
    return result;
  }
  
  Node* Node::findBottomNode(const BARTFit& fit, const double *x) const
  {
    if (isBottom()) return const_cast<Node*>(this);
    
    if (p.rule.goesRight(fit, x)) return p.rightChild->findBottomNode(fit, x);
    
    return leftChild->findBottomNode(fit, x);
  }
}


namespace {
  using namespace dbarts;
  
  struct IndexOrdering {
    const BARTFit& fit;
    const Rule &rule;
    
    IndexOrdering(const BARTFit& fit, const Rule &rule) : fit(fit), rule(rule) { }
    
    bool operator()(size_t i) const { return rule.goesRight(fit, fit.scratch.Xt + i * fit.data.numPredictors); }
  };
  
  // returns how many observations are on the "left"
  size_t partitionRange(size_t* restrict indices, size_t startIndex, size_t length, IndexOrdering& restrict indexGoesRight) {
    size_t lengthOfLeft;
    
    size_t lh = 0, rh = length - 1;
    size_t i = startIndex;
    while (lh <= rh && rh > 0) {
      if (indexGoesRight(i)) {
        indices[rh] = i;
        i = startIndex + rh--;
      } else {
        indices[lh] = i;
        i = startIndex + ++lh;
      }
    }
    if (lh == 0 && rh == 0) { // ugliness w/wrapping around at 0 makes an off-by-one when all obs go right
      indices[startIndex] = i;
      if (indexGoesRight(i)) {
        lengthOfLeft = 0;
      } else {
        lengthOfLeft = 1;
      }
    } else {
      lengthOfLeft = lh;
    }
    return lengthOfLeft;
  }
  
  size_t partitionIndices(size_t* restrict indices, size_t length, IndexOrdering& restrict indexGoesRight) {
    if (length == 0) return 0;
    
    size_t lengthOfLeft;
    
    size_t lh = 0, rh = length - 1;
    while (lh <= rh && rh > 0) {
      if (indexGoesRight(indices[lh])) {
        size_t temp = indices[rh];
        indices[rh] = indices[lh];
        indices[lh] = temp;
        --rh;
      } else {
        ++lh;
      }
    }
    if (lh == 0 && rh == 0) {
      if (indexGoesRight(indices[0])) {
        lengthOfLeft = 0;
      } else {
        lengthOfLeft = 1;
      }
    } else {
      lengthOfLeft = lh;
    }
    
    return lengthOfLeft;
  }
  
  /*
   // http://en.wikipedia.org/wiki/XOR_swap_algorithm
   void ext_swapVectors(size_t* restrict x, size_t* restrict y, size_t length)
   {
   if (length == 0) return;
   
   size_t lengthMod5 = length % 5;
   
   if (lengthMod5 != 0) {
   for (size_t i = 0; i < lengthMod5; ++i) {
   x[i] ^= y[i];
   y[i] ^= x[i];
   x[i] ^= y[i];
   }
   if (length < 5) return;
   }
   
   for (size_t i = lengthMod5; i < length; i += 5) {
   x[i    ] ^= y[i    ]; y[i    ] ^= x[i    ]; x[i    ] ^= y[i    ];
   x[i + 1] ^= y[i + 1]; y[i + 1] ^= x[i + 1]; x[i + 1] ^= y[i + 1];
   x[i + 2] ^= y[i + 2]; y[i + 2] ^= x[i + 2]; x[i + 2] ^= y[i + 2];
   x[i + 3] ^= y[i + 3]; y[i + 3] ^= x[i + 3]; x[i + 3] ^= y[i + 3];
   x[i + 4] ^= y[i + 4]; y[i + 4] ^= x[i + 4]; x[i + 4] ^= y[i + 4];
   }
   }
   
   // merges adjacent partitions of the form:
   // [ l1 r1 l2 r2 ]
   size_t mergeAdjacentPartitions(size_t* array, size_t firstTotalLength, size_t firstLeftLength,
   size_t secondLeftLength)
   {
   // size_t* l1 = array;
   size_t* r1 = array + firstLeftLength;
   size_t* l2 = array + firstTotalLength;
   // size_t* r2 = array + firstTotalLength + secondLeftLength;
   
   size_t firstRightLength = firstTotalLength - firstLeftLength;
   
   if (secondLeftLength <= firstRightLength) {
   ext_swapVectors(r1, l2, secondLeftLength);
   // end up w/[ l1 l2 r1_2 r1_1 r2 ]
   } else {
   ext_swapVectors(r1, l2 + (secondLeftLength - firstRightLength), firstRightLength);
   // end up w/[ l1 l2_2 l2_1 r1 r2 ]
   }
   
   return firstLeftLength + secondLeftLength;
   }
  
  struct PartitionThreadData {
    size_t* indices;
    size_t startIndex;
    size_t length;
    IndexOrdering* ordering;
    size_t numOnLeft;
  };
  
  size_t mergePartitions(PartitionThreadData* data, size_t numThreads)
  {
    while (numThreads > 1) {
      if (numThreads % 2 == 1) {
        // if odd number, merge last two
        PartitionThreadData* left = &data[numThreads - 2];
        PartitionThreadData* right = &data[numThreads - 1];
        
        left->numOnLeft = mergeAdjacentPartitions(left->indices, left->length, left->numOnLeft, right->numOnLeft);
        left->length += right->length;
        
        --numThreads;
      }
        
      for (size_t i = 0; i < numThreads / 2; ++i) {
        PartitionThreadData* left = &data[2 * i];
        PartitionThreadData* right = &data[2 * i + 1];
        
        left->numOnLeft = mergeAdjacentPartitions(left->indices, left->length, left->numOnLeft, right->numOnLeft);
        left->length += right->length;
        
        // now shift down in array so that valid stuffs always occupy the beginning
        if (i > 0) {
          right = &data[i];
          std::memcpy(right, (const PartitionThreadData*) left, sizeof(PartitionThreadData));
        }
      }
      numThreads /= 2;
    }
    return data[0].numOnLeft;
  }
  
  void partitionTask(void* v_data) {
    PartitionThreadData& data(*static_cast<PartitionThreadData*>(v_data));

    data.numOnLeft = (data.startIndex != ((size_t) -1) ? 
                      partitionRange(data.indices, data.startIndex, data.length, *data.ordering) :
                      partitionIndices(data.indices, data.length, *data.ordering));
  } */
} // anon namespace
// MT not worth it for this, apparently
// #define MIN_NUM_OBSERVATIONS_IN_NODE_PER_THREAD 5000

namespace dbarts {
  
#ifdef MATCH_BAYES_TREE
  // This only means something if weights are supplied, which BayesTree didn't have.
  // It is also only meaningful on non-end nodes when using MATCH_BAYES_TREE.

  inline double Node::getNumEffectiveObservations(const BARTFit& fit) const {
    if (fit.data.weights == NULL) return getNumObservations();
    
    if (leftChild == NULL) {
      double n = 0.0;
      if (isTop()) {
        for (size_t i = 0; i < numObservations; ++i) n += fit.data.weights[i];
      } else {
        for (size_t i = 0; i < numObservations; ++i) n += fit.data.weights[observationIndices[i]];
      }
      return n;
    } else {
      return leftChild->getNumEffectiveObservations(fit) + p.rightChild->getNumEffectiveObservations(fit);
    }
  }
#endif
  
  void Node::updateState(const BARTFit& fit, const double* y, uint32_t updateType)
  {
    if (isBottom()) {
      switch(updateType) {
        case BART_NODE_UPDATE_COVARIATES_CHANGED:
        case BART_NODE_UPDATE_COVARIATES_CHANGED | BART_NODE_UPDATE_RESPONSE_PARAMS_CHANGED:
        fit.model.endNodeModel->updateScratchWithMemberships(fit, *this);
        break;
        
        case BART_NODE_UPDATE_VALUES_CHANGED:
        case BART_NODE_UPDATE_VALUES_CHANGED | BART_NODE_UPDATE_RESPONSE_PARAMS_CHANGED:
        leftChild = NULL;
        fit.model.endNodeModel->updateScratchWithValues(fit, *this, y);
        break;
        
        case BART_NODE_UPDATE_COVARIATES_CHANGED | BART_NODE_UPDATE_VALUES_CHANGED:
        case BART_NODE_UPDATE_COVARIATES_CHANGED | BART_NODE_UPDATE_VALUES_CHANGED | BART_NODE_UPDATE_RESPONSE_PARAMS_CHANGED:
        fit.model.endNodeModel->updateScratchWithMembershipsAndValues(fit, *this, y);
        break;
        
        default:
        break;
      }
      return;
    }
    
    // update membership
    if (updateType & BART_NODE_UPDATE_COVARIATES_CHANGED) {
      clearObservationsInNode(fit, *leftChild);
      clearObservationsInNode(fit, *p.rightChild);
  
      size_t numOnLeft = 0;
  
      if (numObservations > 0) {
        IndexOrdering ordering(fit, p.rule);
  
        numOnLeft = (isTop() ?
                     partitionRange(observationIndices, 0, numObservations, ordering) :
                     partitionIndices(observationIndices, numObservations, ordering));
      }
  
  
      leftChild->observationIndices = observationIndices;
      leftChild->numObservations = numOnLeft;
      p.rightChild->observationIndices = observationIndices + numOnLeft;
      p.rightChild->numObservations = numObservations - numOnLeft;
    }
    
    leftChild->updateState(fit, y, updateType);
    p.rightChild->updateState(fit, y, updateType);
  }
  
/*  void Node::updateMembershipsAndValues(const BARTFit& fit, const double* y) {

    if (isBottom()) {
      fit.model.endNodeModel->updateScratchWithMembershipsAndValues(fit, *this, y);
      return;
    }
    
    clearObservationsInNode(fit, *leftChild);
    clearObservationsInNode(fit, *p.rightChild);
    
    size_t numOnLeft = 0;
    
    if (numObservations > 0) {
      IndexOrdering ordering(fit, p.rule);
    
      numOnLeft = (isTop() ?
                   partitionRange(observationIndices, 0, numObservations, ordering) :
                   partitionIndices(observationIndices, numObservations, ordering));
    }
    
    
    leftChild->observationIndices = observationIndices;
    leftChild->numObservations = numOnLeft;
    p.rightChild->observationIndices = observationIndices + numOnLeft;
    p.rightChild->numObservations = numObservations - numOnLeft;
    
    
    leftChild->updateMembershipsAndValues(fit, y);
    p.rightChild->updateMembershipsAndValues(fit, y);
  }
  
  void Node::updateMemberships(const BARTFit& fit) {
    if (isBottom()) {
      fit.model.endNodeModel->updateScratchWithMemberships(fit, *this);
      return;
    }
    
    clearObservationsInNode(fit, *leftChild);
    clearObservationsInNode(fit, *p.rightChild);
    
    size_t numOnLeft = 0;
    if (numObservations > 0) {
      IndexOrdering ordering(fit, p.rule);
    
      numOnLeft = (isTop() ?
                   partitionRange(observationIndices, 0, numObservations, ordering) :
                   partitionIndices(observationIndices, numObservations, ordering));
    }
    
    leftChild->observationIndices = observationIndices;
    leftChild->numObservations = numOnLeft;
    p.rightChild->observationIndices = observationIndices + numOnLeft;
    p.rightChild->numObservations = numObservations - numOnLeft;
    
    leftChild->updateMemberships(fit);
    p.rightChild->updateMemberships(fit);
  }
	
  void Node::updateWithValues(const BARTFit& fit, const double* r)
  {
    leftChild = NULL;
    
    fit.model.endNodeModel->updateScratchWithValues(fit, *this, r);
  }
  
  void Node::updateBottomNodesWithValues(const BARTFit& fit, const double* r)
  {
    if (isBottom()) {
      updateWithValues(fit, r);
      return;
    }
    
    leftChild->updateBottomNodesWithValues(fit, r);
    p.rightChild->updateBottomNodesWithValues(fit, r);
  } */
  
  double Node::drawFromPosterior(const BARTFit& fit, double residualVariance) const
  {
    if (getNumObservations() == 0) return 0.0;
    
    return fit.model.endNodeModel->drawFromPosterior(fit, *this, residualVariance);
  }
  
  // these could potentially be multithreaded, but the gains are probably minimal
  void Node::setPredictions(double* y_hat, double prediction) const
  {
    if (isTop()) {
      ext_setVectorToConstant(y_hat, getNumObservations(), prediction);
      return;
    }
    
    ext_setIndexedVectorToConstant(y_hat, observationIndices, getNumObservations(), prediction);
  }
  
  size_t Node::getDepth() const
  {
    size_t result = 0;
    const Node* node = this;

    while (!node->isTop()) {
      ++result;
      node = node->parent;
    }
    
    return result;
  }
  
  size_t Node::getDepthBelow() const
  {
    if (childrenAreBottom()) return 1;
    if (isBottom()) return 0;
    return (1 + std::max(leftChild->getDepthBelow(), p.rightChild->getDepthBelow()));
  }
  
  size_t Node::getNumNodesBelow() const
  {
    if (isBottom()) return 0;
    return 2 + leftChild->getNumNodesBelow() + p.rightChild->getNumNodesBelow();
  }
  
  size_t Node::getNumVariablesAvailableForSplit(size_t numVariables) const {
    return countTrueValues(variablesAvailableForSplit, numVariables);
  }

  void Node::split(const BARTFit& fit, const Rule& newRule, const double* y, bool exhaustedLeftSplits, bool exhaustedRightSplits) {
    if (newRule.variableIndex < 0) ext_throwError("error in split: rule not set\n");
    
    p.rule = newRule;
    
    leftChild    = Node::create(fit, *this);
    p.rightChild = Node::create(fit, *this);
    
    if (exhaustedLeftSplits)     leftChild->variablesAvailableForSplit[p.rule.variableIndex] = false;
    if (exhaustedRightSplits) p.rightChild->variablesAvailableForSplit[p.rule.variableIndex] = false;
    
    updateState(fit, y, BART_NODE_UPDATE_TREE_STRUCTURE_CHANGED | BART_NODE_UPDATE_VALUES_CHANGED);
    // updateMembershipsAndValues(fit, y);
  }

  void Node::orphanChildren(const BARTFit& fit) {
    fit.model.endNodeModel->updateScratchFromChildren(*this, *leftChild, *p.rightChild);
    
    leftChild = NULL;
  }
  
  void Node::countVariableUses(uint32_t* variableCounts) const
  {
    if (isBottom()) return;
    
    ++variableCounts[p.rule.variableIndex];
    
    leftChild->countVariableUses(variableCounts);
    p.rightChild->countVariableUses(variableCounts);
  }
}
