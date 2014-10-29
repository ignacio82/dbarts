#include "config.hpp"
#include <dbarts/endNodeModel.hpp>

#include <cmath>
#include "dbarts/cstdint.hpp"

#include <external/io.h>
#include <external/random.h>
// #include <external/stats.h>
#include <external/stats_mt.h>

#include <dbarts/bartFit.hpp>
#include <dbarts/control.hpp>
#include "node.hpp"

using std::size_t;
using std::uint32_t;

namespace {
  using namespace dbarts;
  
  struct MeanNormalNodeScratch {
    double average;
    double numEffectiveObservations;
  };
  
  double meanNormalLogIntegratedLikelihood(const EndNode::Model& model, const BARTFit& fit, const Node& node, const double* y, double residualVariance);
  double meanNormalDrawFromPosterior(const EndNode::Model& model, ext_rng* rng, double ybar, double numEffectiveObservations, double residualVariance);
  
  void meanNormalClearScratch(Node& node);
  void meanNormalCopyScratch(Node& target, const Node& source);
  void meanNormalPrintScratch(const Node& node);
  
  void meanNormalUpdateScratchWithResiduals(const BARTFit& fit, const Node& node, const double* r);
}


namespace dbarts {  
  namespace EndNode {
    void initializeMeanNormalModel(MeanNormalModel& model)
    {
      model.info = CONDITIONALLY_INTEGRABLE;
      model.perNodeScratchSize = sizeof(MeanNormalNodeScratch);
      model.precision = 1.0;
      
      model.computeLogIntegratedLikelihood = &meanNormalLogIntegratedLikelihood;
      model.drawFromPosterior = &meanNormalDrawFromPosterior;
      model.clearScratch = &meanNormalClearScratch;
      model.copyScratch = &meanNormalCopyScratch;
      model.printScratch = &meanNormalPrintScratch;
      model.updateScratchWithResiduals = &meanNormalUpdateScratchWithResiduals;
    }
    
    MeanNormalModel* createMeanNormalModel()
    {
      MeanNormalModel* result = new MeanNormalModel;
      initializeMeanNormalModel(*result);
      
      return result;
    }
    
    void initializeMeanNormalModel(MeanNormalModel& model, const Control& control, double k)
    {
      initializeMeanNormalModel(model);
      
      double sigma = (control.responseIsBinary ? 3.0 : 0.5) /  (k * std::sqrt((double) control.numTrees));
      model.precision = 1.0 / (sigma * sigma);
    }
    
    MeanNormalModel* createMeanNormalModel(const Control& control, double k)
    {
      MeanNormalModel* result = createMeanNormalModel();
      initializeMeanNormalModel(*result, control, k);
      
      return result;
    }
  }
}

namespace {
  using dbarts::BARTFit;
  using dbarts::Node;
  
  using dbarts::EndNode::Model;
  using dbarts::EndNode::MeanNormalModel;
  
  double meanNormalDrawFromPosterior(const Model& modelRef, ext_rng* rng, double ybar, double numEffectiveObservations, double residualVariance) {
    const MeanNormalModel& model(static_cast<const MeanNormalModel&>(modelRef));
    
    double posteriorPrecision = numEffectiveObservations / residualVariance;
  
    double posteriorMean = posteriorPrecision * ybar / (model.precision + posteriorPrecision);
    double posteriorSd   = 1.0 / std::sqrt(model.precision + posteriorPrecision);
  
    return posteriorMean + posteriorSd * ext_rng_simulateStandardNormal(rng);
  }
  
  double meanNormalLogIntegratedLikelihood(const Model& modelRef, const BARTFit& fit, const Node& node, const double* y, double residualVariance)
  {
    size_t numObservationsInNode = node.getNumObservations();
    if (numObservationsInNode == 0) return 0.0;
    
    const MeanNormalModel& model(static_cast<const MeanNormalModel&>(modelRef));
    MeanNormalNodeScratch& scratch(*static_cast<MeanNormalNodeScratch*>(node.getScratch()));
    
    double y_bar = scratch.average;
    double var_y = node.computeVariance(fit, y);
      
    double posteriorPrecision = scratch.numEffectiveObservations / residualVariance;
    
    double result;
    result  = 0.5 * std::log(model.precision / (model.precision + posteriorPrecision));
    result -= 0.5 * (var_y / residualVariance) * (double) (numObservationsInNode - 1);
    result -= 0.5 * ((model.precision * y_bar) * (posteriorPrecision * y_bar)) / (model.precision + posteriorPrecision);
    
    return result;
  }
  
  void meanNormalClearScratch(Node& node) {
     MeanNormalNodeScratch& scratch(*static_cast<MeanNormalNodeScratch*>(node.getScratch()));
     scratch.average = 0.0;
  }
  
  void meanNormalCopyScratch(Node& target, const Node& source)
  {
    MeanNormalNodeScratch& targetScratch(*static_cast<MeanNormalNodeScratch*>(target.getScratch()));
    const MeanNormalNodeScratch& sourceScratch(*static_cast<const MeanNormalNodeScratch*>(target.getScratch()));
    
    targetScratch.average = sourceScratch.average;
    targetScratch.numEffectiveObservations = sourceScratch.numEffectiveObservations;
  }
  
  void meanNormalPrintScratch(const Node& node)
  {
    MeanNormalNodeScratch& scratch(*static_cast<MeanNormalNodeScratch*>(node.getScratch()));
    ext_printf(" ave: %f", scratch.average);
  }
  
  void meanNormalUpdateScratchWithResiduals(const BARTFit& fit, const Node& node, const double* r)
  {
    MeanNormalNodeScratch& scratch(*static_cast<MeanNormalNodeScratch*>(node.getScratch()));
        
    size_t numObservations = node.getNumObservations();
    uint32_t updateType = (node.isTop() == false ? 1 : 0) + (fit.data.weights != NULL ? 2 : 0);
    switch (updateType) {
      case 0: // isTop && weights == NULL
      scratch.average = ext_mt_computeMean(fit.threadManager, r, numObservations);
      scratch.numEffectiveObservations = (double) numObservations;
      break;
      
      case 1: // !isTop && weights == NULL
      scratch.average = ext_mt_computeIndexedMean(fit.threadManager, r, node.getObservationIndices(), numObservations);
      scratch.numEffectiveObservations = (double) numObservations;
      break;
      
      case 2: // isTop && weights != NULL
      scratch.average = ext_mt_computeWeightedMean(fit.threadManager, r, numObservations, fit.data.weights, &scratch.numEffectiveObservations);
      break;
      
      case 3: // !isTop && weights != NULL
      scratch.average = ext_mt_computeIndexedWeightedMean(fit.threadManager, r, node.getObservationIndices(), numObservations, fit.data.weights, &scratch.numEffectiveObservations);
      break;
      
      default:
      break;
    }
  }
}
