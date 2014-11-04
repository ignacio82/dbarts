#include "config.hpp"
#include <dbarts/endNodeModel.hpp>

#include <cmath>
#include "dbarts/cstdint.hpp"

#include <external/binaryIO.h>
#include <external/io.h>
#include <external/random.h>
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
  double meanNormalDrawFromPosterior(const EndNode::Model& model, const BARTFit& fit, const Node& node, double residualVariance);
  
  void meanNormalClearScratch(Node& node);
  void meanNormalCopyScratch(Node& target, const Node& source);
  void meanNormalPrintScratch(const Node& node);
  
  void meanNormalUpdateScratchWithValues(const BARTFit& fit, const Node& node, const double* r);
  void meanNormalUpdateScratchFromChildren(Node& parentNode, const Node& leftChildNode, const Node& rightChildNode);
  
  int meanNormalWriteScratch(const Node& node, ext_binaryIO* bio);
  int meanNormalReadScratch(Node& node, ext_binaryIO* bio);
  
  void meanNormalStoreScratch(const Node& node, void* target);
  void meanNormalRestoreScratch(Node& node, const void* source);
  
//   double olsNormalLogIntegratedLikelihood(const EndNode::Model& model, const BARTFit& fit, const Node& node, const double* y, double residualVariance);
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
      model.updateScratchWithValues = &meanNormalUpdateScratchWithValues;
      model.updateScratchWithObservationsAndValues = &meanNormalUpdateScratchWithValues;
      model.updateScratchFromChildren = &meanNormalUpdateScratchFromChildren;
      model.writeScratch = &meanNormalWriteScratch;
      model.readScratch = &meanNormalReadScratch;
      model.storeScratch = &meanNormalStoreScratch;
      model.restoreScratch = &meanNormalRestoreScratch;
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

// normal prior on single mean parameter
namespace {
  using dbarts::BARTFit;
  using dbarts::Node;
  
  using dbarts::EndNode::Model;
  using dbarts::EndNode::MeanNormalModel;
  
  double meanNormalLogIntegratedLikelihood(const Model& modelRef, const BARTFit& fit, const Node& node, const double* y, double residualVariance)
  {
    size_t numObservationsInNode = node.getNumObservations();
    if (numObservationsInNode == 0) return 0.0;
    
    const MeanNormalModel& model(static_cast<const MeanNormalModel&>(modelRef));
    MeanNormalNodeScratch& scratch(*static_cast<MeanNormalNodeScratch*>(node.getScratch()));
    
    double y_bar = scratch.average;
    double var_y = node.computeVariance(fit, y, y_bar);
      
    double dataPrecision = scratch.numEffectiveObservations / residualVariance;
    
    double result;
    result  = 0.5 * std::log(model.precision / (model.precision + dataPrecision));
    result -= 0.5 * (var_y / residualVariance) * (double) (numObservationsInNode - 1);
    result -= 0.5 * ((model.precision * y_bar) * (dataPrecision * y_bar)) / (model.precision + dataPrecision);
    
    return result;
  }
  
  double meanNormalDrawFromPosterior(const Model& modelRef, const BARTFit& fit, const Node& node, double residualVariance) {
    const MeanNormalModel& model(static_cast<const MeanNormalModel&>(modelRef));
    const MeanNormalNodeScratch& scratch(*static_cast<MeanNormalNodeScratch*>(node.getScratch()));
    
    double posteriorPrecision = scratch.numEffectiveObservations / residualVariance;
  
    double posteriorMean = posteriorPrecision * scratch.average / (model.precision + posteriorPrecision);
    double posteriorSd   = 1.0 / std::sqrt(model.precision + posteriorPrecision);
  
    return posteriorMean + posteriorSd * ext_rng_simulateStandardNormal(fit.control.rng);
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
  
  void meanNormalUpdateScratchWithValues(const BARTFit& fit, const Node& node, const double* r)
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
  
  void meanNormalUpdateScratchFromChildren(Node& parentNode, const Node& leftChildNode, const Node& rightChildNode) {
    MeanNormalNodeScratch& parent(*static_cast<MeanNormalNodeScratch*>(parentNode.getScratch()));
    const MeanNormalNodeScratch& leftChild(*static_cast<const MeanNormalNodeScratch*>(leftChildNode.getScratch()));
    const MeanNormalNodeScratch& rightChild(*static_cast<const MeanNormalNodeScratch*>(rightChildNode.getScratch()));

    parent.numEffectiveObservations = leftChild.numEffectiveObservations + rightChild.numEffectiveObservations;    
    parent.average = leftChild.average * (leftChild.numEffectiveObservations / parent.numEffectiveObservations) +
                     rightChild.average * (rightChild.numEffectiveObservations / parent.numEffectiveObservations);
  }
  
  int meanNormalWriteScratch(const Node& node, ext_binaryIO* bio) {
    const MeanNormalNodeScratch& scratch(*static_cast<const MeanNormalNodeScratch*>(node.getScratch()));
    
    int errorCode = ext_bio_writeDouble(bio, scratch.average);
    if (errorCode != 0) return errorCode;
    
    return ext_bio_writeDouble(bio, scratch.numEffectiveObservations);
  }
  
  int meanNormalReadScratch(Node& node, ext_binaryIO* bio) {
    MeanNormalNodeScratch& scratch(*static_cast<MeanNormalNodeScratch*>(node.getScratch()));
    
    int errorCode = ext_bio_readDouble(bio, &scratch.average);
    if (errorCode != 0) return errorCode;
    
    return ext_bio_readDouble(bio, &scratch.numEffectiveObservations);
  }
  
  void meanNormalStoreScratch(const Node& node, void* targetPtr) {
    const MeanNormalNodeScratch& source(*static_cast<const MeanNormalNodeScratch*>(node.getScratch()));
    MeanNormalNodeScratch& target(*static_cast<MeanNormalNodeScratch*>(targetPtr));
    
    target.average = source.average;
    target.numEffectiveObservations = source.numEffectiveObservations;
  }
  
  void meanNormalRestoreScratch(Node& node, const void* sourcePtr) {
    const MeanNormalNodeScratch& source(*static_cast<const MeanNormalNodeScratch*>(sourcePtr));
    MeanNormalNodeScratch& target(*static_cast<MeanNormalNodeScratch*>(node.getScratch()));
    
    target.average = source.average;
    target.numEffectiveObservations = source.numEffectiveObservations;
  }
}

/*
// orindary linear regression with gaussian prior
namespace {
  using dbarts::BARTFit;
  using dbarts::Node;
  
  using dbarts::EndNode::Model;
  using dbarts::EndNode::MeanNormalModel;
  
  double olsNormalLogIntegratedLikelihood(const EndNode::Model& model, const BARTFit& fit, const Node& node, const double* y, double residualVariance)
  {
    
  }
} */
