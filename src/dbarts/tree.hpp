#ifndef DBARTS_TREE_HPP
#define DBARTS_TREE_HPP

#include <cstddef>
#include <dbarts/cstdint.hpp>

#include "node.hpp"

#define TREE_AT(_V_, _I_, _S_) reinterpret_cast<Tree*>(reinterpret_cast<char*>(_V_) + (_I_) * _S_)

struct ext_binaryIO;
struct ext_stringWriter;

namespace dbarts {
  using std::size_t;
  using std::uint32_t;
  
  struct BARTFit;
  struct TreeContentBackup;
  
  struct Tree : Node {
    void drawFromTreeStructurePosterior(const BARTFit& fit, const double* y, double residualVariance);
    void drawFromEndNodePosteriors(const BARTFit& fit, const double* y, double residualVariance);
    void getFits(const BARTFit& fit, const double* y, double* trainingFits, double* testFits);
    
    double* recoverAveragesFromFits(const BARTFit& fit, const double* treeFits); // allocates response; are ordered as bottom nodes are
    void setCurrentFitsFromAverages(const BARTFit& fit, const double* posteriorPredictions, double* trainingFits, double* testFits);
    
    Node* getTop() const;
    bool hasSingleNode() const;
    
    
    void updateBottomNodesWithValues(const BARTFit& fit, const double* y);
    
    const char* createString() const;
    
    int read(const BARTFit& fit, const char* string);
    int read(const BARTFit& fit, ext_binaryIO* bio);
    
    int write(const BARTFit& fit, ext_binaryIO* bio) const;
    int write(ext_stringWriter* writer) const;
    
    bool isValid() const;
  };
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  inline Node* Tree::getTop() const { return const_cast<Node*>(static_cast<const Node*>(this)); }
  inline bool Tree::hasSingleNode() const { return isBottom(); }
}

#endif
