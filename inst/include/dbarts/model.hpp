#ifndef DBARTS_MODEL_HPP
#define DBARTS_MODEL_HPP

#include <cstddef>
#include "cstdint.hpp"

// can make these kinds of adjustments to trees during MCMC
#define DBARTS_BIRTH_OR_DEATH_PROBABILITY 0.5
#define DBARTS_SWAP_PROBABILITY           0.1
#define DBARTS_CHANGE_PROBABILITY         0.4
// conditional on being inside a birthOrDeath step
#define DBARTS_BIRTH_PROBABILITY          0.5

#define DBARTS_DEFAULT_TREE_PRIOR_POWER 2.0
#define DBARTS_DEFAULT_TREE_PRIOR_BASE  0.95

struct ext_rng;

namespace dbarts {
  struct TreePrior;
  namespace EndNode { struct Model; }
  namespace Response { struct Model; }
  
  struct Model {
    double birthOrDeathProbability;
    double swapProbability;
    double changeProbability;

    double birthProbability;
    
    TreePrior* treePrior;
    EndNode::Model* endNodeModel;
    Response::Model* responseModel;
    
    Model() : 
      birthOrDeathProbability(DBARTS_BIRTH_OR_DEATH_PROBABILITY),
      swapProbability(DBARTS_SWAP_PROBABILITY),
      changeProbability(DBARTS_CHANGE_PROBABILITY),
      birthProbability(DBARTS_BIRTH_PROBABILITY), treePrior(NULL), endNodeModel(NULL), responseModel(NULL)
    {
    }
  };
  
  struct BARTFit;
  struct Control;
  struct Node;
  struct Tree;
  struct Rule;
  
  struct TreePrior {
    virtual double computeGrowthProbability(const BARTFit& fit, const Node& node) const = 0;
    virtual double computeTreeLogProbability(const BARTFit& fit, const Tree& tree) const = 0;
    
    virtual double computeSplitVariableLogProbability(const BARTFit& fit, const Node& node) const = 0;
    virtual double computeRuleForVariableLogProbability(const BARTFit& fit, const Node& node) const = 0;
    

    virtual Rule drawRuleAndVariable(const BARTFit& fit, const Node& node, bool* exhaustedLeftSplits, bool* exhaustedRightSplits) const = 0;
    virtual std::int32_t drawSplitVariable(const BARTFit& fit, const Node& node) const = 0;
    virtual Rule drawRuleForVariable(const BARTFit& fit, const Node& node, std::int32_t variableIndex, bool* exhaustedLeftSplits, bool* exhaustedRightSplits) const = 0;
    
    virtual ~TreePrior() { }
  };
  
  // for lack of a better name, calling it the Chipman, George, and McCullough prior
  // Pr(node splits) = base / (1 + depth)^power
  
  struct CGMPrior : TreePrior {
    double base;
    double power;
    
    CGMPrior() { }
    CGMPrior(double base, double power) : base(base), power(power) { }
    virtual ~CGMPrior() { }
    
    virtual double computeGrowthProbability(const BARTFit& fit, const Node& node) const;
    virtual double computeTreeLogProbability(const BARTFit& fit, const Tree& tree) const;
    
    virtual double computeSplitVariableLogProbability(const BARTFit& fit, const Node& node) const;
    virtual double computeRuleForVariableLogProbability(const BARTFit& fit, const Node& node) const;
    
    virtual Rule drawRuleAndVariable(const BARTFit& fit, const Node& node, bool* exhaustedLeftSplits, bool* exhaustedRightSplits) const;
    virtual std::int32_t drawSplitVariable(const BARTFit& fit, const Node& node) const;
    virtual Rule drawRuleForVariable(const BARTFit& fit, const Node& node, std::int32_t variableIndex, bool* exhaustedLeftSplits, bool* exhaustedRightSplits) const;
  };
} // namespace dbarts

#endif // DBARTS_MODEL_HPP
