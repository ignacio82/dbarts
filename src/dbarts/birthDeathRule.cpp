#include "config.hpp"
#include "birthDeathRule.hpp"

#include <cstddef> // size_t
#include <cmath>   // exp
#include <cstring> // memcpy

#include <external/alloca.h>
#include <external/linearAlgebra.h>
#include <external/random.h>

#include <dbarts/bartFit.hpp>
#include <dbarts/model.hpp>
#include "likelihood.hpp"
#include "node.hpp"
#include "tree.hpp"

using std::size_t;


namespace {
  using namespace dbarts;
  
  void acceptBirth(const BARTFit& fit, Node& copyOfOriginalNode);
  void rejectBirth(const BARTFit& fit, const Node& copyOfOriginalNode, Node& modifiedNode);
  
  void acceptDeath(const BARTFit& fit, Node& copyOfOriginalNode);
  void rejectDeath(const BARTFit& fit, const Node& copyOfOriginalNode, Node& modifiedNode);
}

namespace dbarts {
  
  Node* drawBirthableNode(const BARTFit& fit, const Tree& tree, double* nodeSelectionProbability);
  Node* drawChildrenKillableNode(const BARTFit& fit, const Tree& tree, double* nodeSelectionProbability);
  
  double computeUnnormalizedNodeBirthProbability(const BARTFit& fit, const Node& node);
  double computeProbabilityOfBirthStep(const BARTFit& fit, const Tree& tree); // same as below but that has a step cached
  double computeProbabilityOfBirthStep(const BARTFit& fit, const Tree& tree, bool birthableNodeExists);
  double computeProbabilityOfSelectingNodeForDeath(const Tree& tree);
  double computeProbabilityOfSelectingNodeForBirth(const BARTFit& fit, const Tree& tree);
    
  // returns probability of jump
  double birthOrDeathNode(const BARTFit& fit, Tree& tree, const double* y, bool* stepWasTaken, bool* stepWasBirth)
  {
    double ratio;
    
#ifdef alloca
    Node* originalNodePtr = static_cast<Node*>(alloca(fit.scratch.nodeSize));
#else
    Node* originalNodePtr = static_cast<Node*>(::operator new(fit.scratch.nodeSize));
#endif
    Node& copyOfOriginalNode(*originalNodePtr);
    
    // Rather than flipping a coin to see if birth or death, we have to first check that either is possible.
    // Since that involves pretty much finding a node to give birth, we just do that and then possibly ignore
    // it.
    
    double sigma_sq = fit.state.sigma * fit.state.sigma;

    double transitionProbabilityOfSelectingNodeForBirth;
    Node* nodeToChangePtr = drawBirthableNode(fit, tree, &transitionProbabilityOfSelectingNodeForBirth);
    
    double transitionProbabilityOfBirthStep = computeProbabilityOfBirthStep(fit, tree, nodeToChangePtr != NULL);
    
    if (ext_rng_simulateBernoulli(fit.control.rng, transitionProbabilityOfBirthStep) == 1) {
      *stepWasBirth = true;
      
      Node& nodeToChange(*nodeToChangePtr);
      
      double parentPriorGrowthProbability = fit.model.treePrior->computeGrowthProbability(fit, nodeToChange);
      double oldPriorProbability = 1.0 - parentPriorGrowthProbability;
      double oldLogLikelihood = computeLogLikelihoodForBranch(fit, nodeToChange, y);
      
      // now perform birth;
      // shallow copy
      std::memcpy(&copyOfOriginalNode, const_cast<Node*>(&nodeToChange), fit.scratch.nodeSize);
      // copyOfOriginalNode.copyFrom(fit, nodeToChange);

      bool exhaustedLeftSplits, exhaustedRightSplits;
      Rule newRule = fit.model.treePrior->drawRuleAndVariable(fit, nodeToChange, &exhaustedLeftSplits, &exhaustedRightSplits);
      nodeToChange.split(fit, newRule, y, sigma_sq, exhaustedLeftSplits, exhaustedRightSplits);
      
      // determine how to go backwards
      double leftPriorGrowthProbability  = fit.model.treePrior->computeGrowthProbability(fit, *nodeToChange.getLeftChild());
      double rightPriorGrowthProbability = fit.model.treePrior->computeGrowthProbability(fit, *nodeToChange.getRightChild());
      double newPriorProbability = parentPriorGrowthProbability * (1.0 - leftPriorGrowthProbability) * (1.0 - rightPriorGrowthProbability);

      double newLogLikelihood = computeLogLikelihoodForBranch(fit, nodeToChange, y);

      double transitionProbabilityOfDeathStep = 1.0 - computeProbabilityOfBirthStep(fit, tree);
      double transitionProbabilityOfSelectingNodeForDeath = computeProbabilityOfSelectingNodeForDeath(tree);
      
      // compute ratios
      double priorRatio = newPriorProbability / oldPriorProbability;
      double transitionRatio = (transitionProbabilityOfDeathStep * transitionProbabilityOfSelectingNodeForDeath) /
                               (transitionProbabilityOfBirthStep * transitionProbabilityOfSelectingNodeForBirth);
      
      double likelihoodRatio = std::exp(newLogLikelihood - oldLogLikelihood);
      
      ratio = priorRatio * likelihoodRatio * transitionRatio;
      
      if (ext_rng_simulateContinuousUniform(fit.control.rng) < ratio) {
        acceptBirth(fit, copyOfOriginalNode);
        
        *stepWasTaken = true;
      } else {
        rejectBirth(fit, const_cast<Node&>(copyOfOriginalNode), nodeToChange);
        
        *stepWasTaken = false;
      }
    } else {
      *stepWasBirth = false;
      
      double transitionProbabilityOfDeathStep = 1.0 - transitionProbabilityOfBirthStep;
      
      double transitionProbabilityOfSelectingNodeForDeath;
      nodeToChangePtr = drawChildrenKillableNode(fit, tree, &transitionProbabilityOfSelectingNodeForDeath);
      
      Node& nodeToChange(*nodeToChangePtr);
      
      double parentPriorGrowthProbability = fit.model.treePrior->computeGrowthProbability(fit, nodeToChange);
      double leftPriorGrowthProbability   = fit.model.treePrior->computeGrowthProbability(fit, *nodeToChange.getLeftChild());
      double rightPriorGrowthProbability  = fit.model.treePrior->computeGrowthProbability(fit, *nodeToChange.getRightChild());
      double oldLogLikelihood = computeLogLikelihoodForBranch(fit, nodeToChange, y);
      
      std::memcpy(&copyOfOriginalNode, const_cast<Node*>(&nodeToChange), fit.scratch.nodeSize);
      //copyOfOriginalNode.copyFrom(nodeToChange);
      
      // now figure out how the node could have given birth
      nodeToChange.orphanChildren(fit, y, sigma_sq);
      
      double newLogLikelihood = computeLogLikelihoodForBranch(fit, nodeToChange, y);
      transitionProbabilityOfBirthStep = computeProbabilityOfBirthStep(fit, tree, true);
#ifdef MATCH_BAYES_TREE
      ext_simulateContinuousUniform();
#endif
      double transitionProbabilityOfSelectingNodeForBirth = computeProbabilityOfSelectingNodeForBirth(fit, tree);
      
      double oldPriorProbability = parentPriorGrowthProbability * (1.0 - leftPriorGrowthProbability) * (1.0 - rightPriorGrowthProbability);
      double newPriorProbability = 1.0 - parentPriorGrowthProbability;
      
      double priorRatio = newPriorProbability / oldPriorProbability;
      double transitionRatio = (transitionProbabilityOfBirthStep * transitionProbabilityOfSelectingNodeForBirth) /
                               (transitionProbabilityOfDeathStep * transitionProbabilityOfSelectingNodeForDeath);
      
      double likelihoodRatio = std::exp(newLogLikelihood - oldLogLikelihood);
      
      ratio = priorRatio * likelihoodRatio * transitionRatio;
      
      if (ext_rng_simulateContinuousUniform(fit.control.rng) < ratio) {
        acceptDeath(fit, copyOfOriginalNode);
        
        *stepWasTaken = true;
      } else {
        rejectDeath(fit, const_cast<Node&>(copyOfOriginalNode), nodeToChange);
        
        *stepWasTaken = false;
      }
    }
    
#ifndef alloca
    ::operator delete(originalNodePtr);
#endif
    
    return ratio < 1.0 ? ratio : 1.0;
  }
  
  // transition mechanism
  double computeProbabilityOfBirthStep(const BARTFit& fit, const Tree& tree)
  {
    NodeVector bottomNodes(tree.getBottomVector());
    size_t numBottomNodes = bottomNodes.size();
    
    bool birthableNodeExists = false;
    
    for (size_t i = 0; i < numBottomNodes; ++i) {
      if (computeUnnormalizedNodeBirthProbability(fit, *bottomNodes[i]) > 0.0) {
        birthableNodeExists = true;
        break;
      }
    }
    
#ifdef MATCH_BAYES_TREE
    if (birthableNodeExists) ext_simulateContinuousUniform();
#endif
    
    return computeProbabilityOfBirthStep(fit, tree, birthableNodeExists);
  }
  
  double computeProbabilityOfBirthStep(const BARTFit& fit, const Tree& tree, bool birthableNodeExists)
  {
    if (!birthableNodeExists) return 0.0;
    if (tree.hasSingleNode()) return 1.0;
    
    return fit.model.birthProbability;
  }
  
  double computeProbabilityOfSelectingNodeForDeath(const Tree& tree)
  {
    size_t numNodesWhoseChildrenAreBottom = tree.getNumNoGrandNodes();
    if (numNodesWhoseChildrenAreBottom == 0) return 0.0;
    
    return 1.0 / static_cast<double>(numNodesWhoseChildrenAreBottom);
  }
                                                                                                      
  double computeProbabilityOfSelectingNodeForBirth(const BARTFit& fit, const Tree& tree)
  {
    if (tree.hasSingleNode()) return 1.0;
    
    NodeVector bottomNodes(tree.getBottomVector());
    size_t numBottomNodes = bottomNodes.size();
    
    double totalProbability = 0.0;
    
    for (size_t i = 0; i < numBottomNodes; ++i) {
      totalProbability += computeUnnormalizedNodeBirthProbability(fit, *bottomNodes[i]);
    }
    
    if (totalProbability <= 0.0) return 0.0;
    
    return 1.0 / totalProbability;
  }
  
  Node* drawBirthableNode(const BARTFit& fit, const Tree& tree, double* nodeSelectionProbability)
  {
    Node* result = NULL;
    
#ifndef MATCH_BAYES_TREE
    if (tree.hasSingleNode()) {
      *nodeSelectionProbability = 1.0;
      return tree.getTop();
    }
#endif
    
    NodeVector bottomNodes(tree.getBottomVector());
    size_t numBottomNodes = bottomNodes.size();
    
    double* nodeBirthProbabilities = ext_stackAllocate(numBottomNodes, double);
    double totalProbability = 0.0;
        
    for (size_t i = 0; i < numBottomNodes; ++i) {
      nodeBirthProbabilities[i] = computeUnnormalizedNodeBirthProbability(fit, *bottomNodes[i]);
      totalProbability += nodeBirthProbabilities[i];
    }
    
    if (totalProbability > 0.0) {
      ext_scalarMultiplyVectorInPlace(nodeBirthProbabilities, numBottomNodes, 1.0 / totalProbability);

      size_t index = ext_rng_drawFromDiscreteDistribution(fit.control.rng, nodeBirthProbabilities, numBottomNodes);

      result = bottomNodes[index];
      *nodeSelectionProbability = nodeBirthProbabilities[index];
    } else {
      *nodeSelectionProbability = 0.0;
    }
    
    ext_stackFree(nodeBirthProbabilities);
    
    return result;
  }
  
  Node* drawChildrenKillableNode(const BARTFit& fit, const Tree& tree, double* nodeSelectionProbability)
  {
    NodeVector nodesWhoseChildrenAreBottom(tree.getNoGrandVector());
    size_t numNodesWhoseChildrenAreBottom = nodesWhoseChildrenAreBottom.size();
    
    if (numNodesWhoseChildrenAreBottom == 0) {
      *nodeSelectionProbability = 0.0;
      return NULL;
    }
    
    size_t index = ext_rng_simulateUnsignedIntegerUniformInRange(fit.control.rng, 0, numNodesWhoseChildrenAreBottom);
    *nodeSelectionProbability = 1.0 / static_cast<double>(numNodesWhoseChildrenAreBottom);
    
    return nodesWhoseChildrenAreBottom[index];
  }
  
  double computeUnnormalizedNodeBirthProbability(const BARTFit& fit, const Node& node)
  {
    bool hasVariablesAvailable = node.getNumVariablesAvailableForSplit(fit.data.numPredictors) > 0;
    
    return hasVariablesAvailable ? 1.0 : 0.0;
  }
}

namespace {
  using namespace dbarts;
  
  void acceptBirth(const BARTFit& fit, Node& copyOfOriginalNode)
  {
    // no longer an end-node
    if (fit.model.endNodeModel->destroyScratch != NULL) fit.model.endNodeModel->destroyScratch(fit, copyOfOriginalNode.getScratch());
  }
  
  void rejectBirth(const BARTFit& fit, const Node& copyOfOriginalNode, Node& modifiedNode)
  {
    Node::destroy(fit, modifiedNode.getLeftChild());
    Node::destroy(fit, modifiedNode.getRightChild());
    std::memcpy(&modifiedNode, &copyOfOriginalNode, fit.scratch.nodeSize);
    // modifiedNode.copyFrom(fit, copyOfOriginalNode);
  }
  
  void acceptDeath(const BARTFit& fit, Node& copyOfOriginalNode)
  {
    Node::destroy(fit, copyOfOriginalNode.getLeftChild());
    Node::destroy(fit, copyOfOriginalNode.getRightChild());
  }
  
  void rejectDeath(const BARTFit& fit, const Node& copyOfOriginalNode, Node& modifiedNode)
  {
    // no longer an end-node
    if (fit.model.endNodeModel->destroyScratch != NULL) fit.model.endNodeModel->destroyScratch(fit, modifiedNode.getScratch());
    std::memcpy(&modifiedNode, &copyOfOriginalNode, fit.scratch.nodeSize);
    // modifiedNode.copyFrom(fit, copyOfOriginalNode);
  }
}
