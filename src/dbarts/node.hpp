#ifndef DBARTS_NODE_HPP
#define DBARTS_NODE_HPP

#include <dbarts/cstdint.hpp>
#include <cstddef>
#include <vector>

#include <dbarts/types.hpp>

struct ext_rng;

#define NODE_AT(_V_, _I_, _S_) ((Node*) ((char*) (_V_) + (_I_) * _S_))


namespace dbarts {
  using std::size_t;
  using std::uint32_t;
  
  struct BARTFit;
//  struct EndNodePrior;
  
  namespace EndNode { struct Model; }
  
#define DBARTS_INVALID_RULE_VARIABLE -1
  struct Rule {
    int32_t variableIndex;
    
    union {
      int32_t splitIndex;
      uint32_t categoryDirections;
    };
    
    void invalidate();
    
    bool goesRight(const BARTFit& fit, const double* x) const;
    bool categoryGoesRight(uint32_t categoryId) const;
    void setCategoryGoesRight(uint32_t categoryId);
    void setCategoryGoesLeft(uint32_t categoryId);
    double getSplitValue(const BARTFit& fit) const;
    
    bool equals(const Rule& other) const;
    void copyFrom(const Rule& other);
    void swapWith(Rule& other);
  };
  
  struct VarUsage {
    uint32_t depth;
    size_t nodeIndex;
    uint32_t variableIndex;
  };
  
  struct Node;
  
  struct ParentMembers {
    Node* rightChild;
    Rule rule;
  };
  
  struct EndNodeMembers {
    double average;
    double numEffectiveObservations;
  };
  
  typedef std::vector<Node*> NodeVector;
  
  Node* createNode(const BARTFit& fit, size_t* observationIndices, size_t numObservations);
  Node* createNode(const BARTFit& fit, const Node& parent);
  void initializeNode(Node& node, const BARTFit& fit, size_t* observationIndices, size_t numObservations);
  
  void deleteNode(Node* node);
  void invalidateNode(Node& node);
  
  
  struct Node {
    Node* parent;
    Node* leftChild;
    
//    union {
      ParentMembers p;
//      EndNodeMembers m;
//    };
    
#define BART_INVALID_NODE_ENUM ((size_t) -1)
    size_t enumerationIndex;
    bool* variablesAvailableForSplit;
    
    size_t* observationIndices;
    size_t numObservations;
    
//    Node(size_t* observationIndices, size_t numObservations, size_t numPredictors); // node is assumed at top
//    Node(const Node& parent, size_t numPredictors); // node attaches to parent; parent should add observations
//    ~Node();
    
    void copyFrom(const BARTFit& fit, const Node& other);
    
    
    bool isTop() const;
    bool isBottom() const;
    bool childrenAreBottom() const;
    
    Node* getParent() const;
    Node* getLeftChild() const;
    Node* getRightChild() const;
    
    void* getScratch() const;
    
    size_t getNumBottomNodes() const;
    size_t getNumNotBottomNodes() const;
    size_t getNumNoGrandNodes() const;
    size_t getNumSwappableNodes() const;
    
    NodeVector getBottomVector() const;
    NodeVector getNoGrandVector() const;
    NodeVector getNotBottomVector() const;
    NodeVector getSwappableVector() const;
    
    NodeVector getAndEnumerateBottomVector(); // the nodes will have their enumeration indices set to their array index
    
    Node* findBottomNode(const BARTFit& fit, const double* x) const;
        
    void print(const BARTFit& fit) const;
    
    void setAverage(double average);                       // call these only on bottom nodes
    void updateScratchWithResiduals(const BARTFit& fit, const double* r);
    void updateScratchesWithResiduals(const BARTFit& fit, const double* r); // call anywhere and it'll recurse
    void setNumEffectiveObservations(double n);
    
    double getAverage() const;
    double getNumEffectiveObservations() const;
    double computeVariance(const BARTFit& fit, const double* y) const;
    
    size_t getNumObservations() const;
    const size_t* getObservationIndices() const;
    
    void addObservationsToChildren(const BARTFit& fit);
    void addObservationsToChildren(const BARTFit& fit, const double* y); // computes averages in bottom nodes as it goes
    void clearObservations(const BARTFit& fit);
    void clear(const BARTFit& fit);
    
    // double drawFromPosterior(ext_rng* rng, const EndNodePrior& endNodePrior, double residualVariance) const;
    double drawFromPosterior(ext_rng* rng, const EndNode::Model& endNodeModel, double residualVariance) const;
    void setPredictions(double* y_hat, double prediction) const;
        
    size_t getDepth() const;
    size_t getDepthBelow() const;
    
    size_t getNumNodesBelow() const;
    size_t getNumVariablesAvailableForSplit(size_t numVariables) const;
    
    void split(const BARTFit& fit, const Rule& rule, const double* y, bool exhaustedLeftSplits, bool exhaustedRightSplits);
    void orphanChildren();
    
    void countVariableUses(uint32_t* variableCounts) const;
  };
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  // here for inlining purposes but contain implementation details and shouldn't be
  // relied on
  inline bool Node::isTop() const { return parent == NULL; }
  inline bool Node::isBottom() const { return leftChild == NULL; }
  inline bool Node::childrenAreBottom() const { return leftChild != NULL && leftChild->leftChild == NULL && p.rightChild->leftChild == NULL; }
  
  inline Node* Node::getParent() const { return const_cast<Node*>(parent); }
  inline Node* Node::getLeftChild() const { return const_cast<Node*>(leftChild); }
  inline Node* Node::getRightChild() const { return const_cast<Node*>(p.rightChild); }
  
  inline void* Node::getScratch() const { return (void*) &p; }

  inline size_t Node::getNumObservations() const { return numObservations; }
  inline const size_t* Node::getObservationIndices() const { return observationIndices; }
//  inline double Node::getAverage() const { return m.average; }
  inline double Node::getAverage() const { return ((EndNodeMembers*) &p)->average; }

#ifdef MATCH_BAYES_TREE
  // This only means something if weights are supplied, which BayesTree didn't have.
  // It is also only meaningful on non-end nodes when using MATCH_BAYES_TREE.
//  inline double Node::getNumEffectiveObservations() const { if (leftChild == NULL) return m.numEffectiveObservations; else return leftChild->getNumEffectiveObservations() + p.rightChild->getNumEffectiveObservations(); }
  inline double Node::getNumEffectiveObservations() const { if (leftChild == NULL) return ((EndNodeMembers*) &p)->numEffectiveObservations; else return leftChild->getNumEffectiveObservations() + p.rightChild->getNumEffectiveObservations(); }
#else
//  inline double Node::getNumEffectiveObservations() const { return m.numEffectiveObservations; }
  inline double Node::getNumEffectiveObservations() const { return ((EndNodeMembers*) &p)->numEffectiveObservations; }
#endif
  // inline void Node::setAverage(double newAverage) { leftChild = NULL; m.average = newAverage; }
  // inline void Node::setNumEffectiveObservations(double n) { leftChild = NULL; m.numEffectiveObservations = n; }
  inline void Node::setAverage(double newAverage) { leftChild = NULL; ((EndNodeMembers*) &p)->average = newAverage; }
  inline void Node::setNumEffectiveObservations(double n) { leftChild = NULL; ((EndNodeMembers*) &p)->numEffectiveObservations = n; }
  
  
  inline bool Rule::categoryGoesRight(uint32_t categoryId) const { return ((1u << categoryId) & categoryDirections) != 0; }
  inline void Rule::setCategoryGoesRight(uint32_t categoryId) { categoryDirections |= (1u << categoryId); }
  inline void Rule::setCategoryGoesLeft(uint32_t categoryId) { categoryDirections &= ~(1u << categoryId); }
}

#endif
