#ifndef DBARTS_NODE_HPP
#define DBARTS_NODE_HPP

#include <dbarts/cstdint.hpp>
#include <cstddef>
#include <vector>

#include <dbarts/types.hpp>

#define NODE_AT(_V_, _I_, _S_) reinterpret_cast<Node*>(reinterpret_cast<char*>(_V_) + (_I_) * _S_)

namespace dbarts {
  using std::size_t;
  using std::uint32_t;
  
  struct BARTFit;
  
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
  
  typedef std::vector<Node*> NodeVector;

  struct Node {
    Node* parent;
    Node* leftChild;
    
#define BART_INVALID_NODE_ENUM static_cast<size_t>(-1)

    size_t enumerationIndex;
    bool* variablesAvailableForSplit;
    
    size_t* observationIndices;
    size_t numObservations;
    
    ParentMembers p; // has to be last member; is actually a union
    
    // these are static because the size is actually determined by fit, and thus calling a constructor
    // won't allocate the correct amount
    // static Node* create(const BARTFit& fit, size_t* observationIndices, size_t numObservations);
    static Node* create(const BARTFit& fit, const Node& parent);
    
    static void initialize(const BARTFit& fit, Node& node, size_t* observationIndices, size_t numObservations);
    static void destroy(const BARTFit& fit, Node* node); // 'delete' reserved
    static void invalidate(const BARTFit& fit, Node& node);
    
#define BART_NODE_UPDATE_COVARIATES_CHANGED      0x1 // top two do the same thing, i.e. calculate memberships
#define BART_NODE_UPDATE_TREE_STRUCTURE_CHANGED  0x1
#define BART_NODE_UPDATE_VALUES_CHANGED          0x2
#define BART_NODE_UPDATE_RESPONSE_PARAMS_CHANGED 0x4
    void updateState(const BARTFit& fit, const double* y, uint32_t updateType);
    
/*     void updateWithValues(const BARTFit& fit, const double* y); // call this only on a bottom node
    void updateBottomNodesWithValues(const BARTFit& fit, const double* y); // call anywhere and it'll recurse
    
    void updateMemberships(const BARTFit& fit);
    void updateMembershipsAndValues(const BARTFit& fit, const double* y); */
    
    // deep copies
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
        
//    double computeVariance(const BARTFit& fit, const double* y, double average) const;
    
    size_t getNumObservations() const;
    const size_t* getObservationIndices() const;
#ifdef MATCH_BAYES_TREE
    double getNumEffectiveObservations(const BARTFit& fit) const;
#endif
    
    
    void clear(const BARTFit& fit);
    
    double drawFromPosterior(const BARTFit& fit, double residualVariance) const;
    void setPredictions(double* y_hat, double prediction) const;
        
    size_t getDepth() const;
    size_t getDepthBelow() const;
    
    size_t getNumNodesBelow() const;
    size_t getNumVariablesAvailableForSplit(size_t numVariables) const;
    
    // split and orphanChildren *DO NOT* handle node scratch, so that they are more readily reversable
    void split(const BARTFit& fit, const Rule& rule, const double* y, bool exhaustedLeftSplits, bool exhaustedRightSplits);
    void orphanChildren(const BARTFit& fit);
    
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
  
  inline void* Node::getScratch() const { return reinterpret_cast<void*>(const_cast<ParentMembers*>(&p)); }

  inline size_t Node::getNumObservations() const { return numObservations; }
  inline const size_t* Node::getObservationIndices() const { return observationIndices; }

  inline bool Rule::categoryGoesRight(uint32_t categoryId) const { return ((1u << categoryId) & categoryDirections) != 0; }
  inline void Rule::setCategoryGoesRight(uint32_t categoryId) { categoryDirections |= (1u << categoryId); }
  inline void Rule::setCategoryGoesLeft(uint32_t categoryId) { categoryDirections &= ~(1u << categoryId); }
}

#endif
