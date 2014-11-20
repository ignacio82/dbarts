#include "config.hpp"
#include "likelihood.hpp"

#include <cstddef>

#include <dbarts/bartFit.hpp>
#include <dbarts/endNodeModel.hpp>
#include <dbarts/responseModel.hpp>
#include <dbarts/state.hpp>
#include "node.hpp"

namespace dbarts {
  using std::size_t;
  
  double computeLogLikelihoodForBranch(const BARTFit& fit, const Node& branch, const double* y)
  {
    NodeVector bottomVector(branch.getBottomVector());
    size_t numBottomNodes = bottomVector.size();
    
    double logProbability = 0.0;
    double sigma_sq = static_cast<const Response::NormalChiSquaredModel*>(fit.model.responseModel)->sigma;
    sigma_sq *= sigma_sq;
    
    for (size_t i = 0; i < numBottomNodes; ++i) {
      const Node& bottomNode(*bottomVector[i]);
      
      if (bottomNode.getNumObservations() == 0) return -10000000.0;
      
      logProbability += fit.model.endNodeModel->computeLogIntegratedLikelihood(fit, bottomNode, y, sigma_sq);
    }
    
    return logProbability;
  }
}
