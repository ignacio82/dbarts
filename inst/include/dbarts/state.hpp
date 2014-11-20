#ifndef DBARTS_STATE_HPP
#define DBARTS_STATE_HPP

#include <cstddef>

namespace dbarts {
  struct BARTFit;
  
  struct State {
    void* trees;
    std::size_t* treeIndices; // numObs * numTree
    
    double* treeFits;      // numObs x numTrees;     vals for tree <=> x + i * numObs
    double* totalFits;
    double* totalTestFits; // numTestObs x 1

    double runningTime;
    
    char** createTreeStructuresStrings(const BARTFit& fit) const;
    void recreateTreeStructuresFromStrings(const BARTFit& fit, const char** treeStructuresStrings);
    
    double** createTreeParametersVectors(const BARTFit& fit, size_t** treeParameterLengths) const;
    void setTreeParametersFromVectors(const BARTFit& fit, double** treeParameters);
    
    void setTreeParametersFromFits(const BARTFit& fit);
  };
} // namespace dbarts

#endif // DBARTS_STATE_HPP
