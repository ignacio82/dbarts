#include "config.hpp"
#include <dbarts/endNodeModel.hpp>

#include <cmath>
#include "dbarts/cstdint.hpp"
#include <cstring>

#include <external/alloca.h>
#include <external/binaryIO.h>
#include <external/io.h>
#include <external/linearAlgebra.h>
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
    double mu; // this plays two roles: average of ys when calculating integrated log-like and posterior draw of node-mean
    double numEffectiveObservations;
  };
  
  double meanNormalLogPrior(const BARTFit& fit, const Node& node);
  double meanNormalLogIntegratedLikelihood(const BARTFit& fit, const Node& node, const double* y, double residualVariance);
  void meanNormalDrawFromPrior(const BARTFit& fit, const Node& node);
  void meanNormalDrawFromPosterior(const BARTFit& fit, const Node& node, const double* y, double residualVariance);
  
  double meanNormalGetPrediction(const BARTFit& fit, const Node& node, const double* y, const double* Xt);
  void meanNormalGetPredictions(const BARTFit& fit, const Node& node, const double* y, double* y_hat);
  void meanNormalGetPredictionsForIndices(const BARTFit& fit, const Node& node, const double* y, const size_t* indices, double* y_hat);
  
  void meanNormalCopyScratch(const BARTFit& fit, Node& target, const Node& source);
  void meanNormalPrintScratch(const BARTFit& fit, const Node& node);
  
  void meanNormalUpdateScratchWithValues(const BARTFit& fit, const Node& node, const double* y);
  void meanNormalUpdateScratchWithMemberships(const BARTFit& fit, const Node& node);
  void meanNormalUpdateScratchFromChildren(const BARTFit& fit, Node& parent, const double* y, const Node& leftChild, const Node& rightChild);
  
  int meanNormalWriteScratch(const Node& node, ext_binaryIO* bio);
  int meanNormalReadScratch(Node& node, ext_binaryIO* bio);
  
  void meanNormalStoreScratch(const Node& node, void* target);
  void meanNormalRestoreScratch(Node& node, const void* source);
}


namespace dbarts {  
  namespace EndNode {
    void initializeMeanNormalModel(MeanNormalModel& model)
    {
      model.info = CONDITIONALLY_INTEGRABLE | PREDICTION_IS_CONSTANT;
      model.perNodeScratchSize = sizeof(MeanNormalNodeScratch);
      model.precision = 1.0;
      
      model.computeLogPrior = &meanNormalLogPrior;
      model.computeLogIntegratedLikelihood = &meanNormalLogIntegratedLikelihood;
      model.drawFromPrior = &meanNormalDrawFromPrior;
      model.drawFromPosterior = &meanNormalDrawFromPosterior;
      
      model.getPrediction = &meanNormalGetPrediction;
      model.getPredictions = &meanNormalGetPredictions;
      model.getPredictionsForIndices = &meanNormalGetPredictionsForIndices;
      
      model.createScratch = NULL;
      model.destroyScratch = NULL;
      model.copyScratch = &meanNormalCopyScratch;
      model.printScratch = &meanNormalPrintScratch;
      
      model.updateScratchWithValues = &meanNormalUpdateScratchWithValues;
      model.updateScratchWithMemberships = &meanNormalUpdateScratchWithMemberships;
      model.updateScratchWithMembershipsAndValues = &meanNormalUpdateScratchWithValues;
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
      
      double sigma = (control.responseIsBinary ? 3.0 : 0.5) /  (k * std::sqrt(static_cast<double>(control.numTrees)));
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
  
#define DEFINE_MODEL(_FIT_) const MeanNormalModel& model(*static_cast<const MeanNormalModel*>(_FIT_.model.endNodeModel));
#define DEFINE_SCRATCH(_NODE_) MeanNormalNodeScratch& scratch(*static_cast<MeanNormalNodeScratch*>(_NODE_.getScratch()));
  
  using dbarts::BARTFit;
  using dbarts::Node;
  
  using dbarts::EndNode::MeanNormalModel;
  
  double computeVarianceForNode(const BARTFit& fit, const Node& node, const double* y, double average)
  {
    size_t numObservations = node.getNumObservations();
    uint32_t updateType = (node.isTop() == false ? 1 : 0) + (fit.data.weights != NULL ? 2 : 0);
    switch (updateType) {
      case 0: // isTop && weights == NULL
      return ext_mt_computeVarianceForKnownMean(fit.threadManager, y, numObservations, average);
      
      case 1: // !isTop && weights == NULL
      return ext_mt_computeIndexedVarianceForKnownMean(fit.threadManager, y, node.getObservationIndices(), numObservations, average);
      
      case 2: // isTop && weights != NULL
      return ext_mt_computeWeightedVarianceForKnownMean(fit.threadManager, y, numObservations, fit.data.weights, average);
      
      case 3: // !isTop && weights != NULL
      return ext_mt_computeIndexedWeightedVarianceForKnownMean(fit.threadManager, y, node.getObservationIndices(), numObservations, fit.data.weights, average);
      
      default:
      break;
    }
    
    return NAN;
  }
  
  double meanNormalLogPrior(const BARTFit& fit, const Node& node)
  {
    DEFINE_MODEL(fit);
    DEFINE_SCRATCH(node);
    
    return -0.5 * scratch.mu * scratch.mu * model.precision;
  }
  
  double meanNormalLogIntegratedLikelihood(const BARTFit& fit, const Node& node, const double* y, double residualVariance)
  {
    size_t numObservationsInNode = node.getNumObservations();
    if (numObservationsInNode == 0) return 0.0;
    
    DEFINE_MODEL(fit);
    DEFINE_SCRATCH(node);
    
    double y_bar = scratch.mu;
    double var_y = computeVarianceForNode(fit, node, y, y_bar);
      
    double dataPrecision = scratch.numEffectiveObservations / residualVariance;
    
    double result;
    result  = 0.5 * std::log(model.precision / (model.precision + dataPrecision));
    result -= 0.5 * (var_y / residualVariance) * static_cast<double>(numObservationsInNode - 1);
    result -= 0.5 * ((model.precision * y_bar) * (dataPrecision * y_bar)) / (model.precision + dataPrecision);
    
    return result;
  }
  
  void meanNormalDrawFromPrior(const BARTFit& fit, const Node& node)
  {
    DEFINE_MODEL(fit);
    DEFINE_SCRATCH(node);
    
    scratch.mu = ext_rng_simulateStandardNormal(fit.control.rng) / std::sqrt(model.precision);
  }
  
  void meanNormalDrawFromPosterior(const BARTFit& fit, const Node& node, const double*, double residualVariance) {
    DEFINE_MODEL(fit);
    DEFINE_SCRATCH(node);
    
    double posteriorPrecision = scratch.numEffectiveObservations / residualVariance;
  
    double posteriorMean = posteriorPrecision * scratch.mu / (model.precision + posteriorPrecision);
    double posteriorSd   = 1.0 / std::sqrt(model.precision + posteriorPrecision);
  
    scratch.mu = posteriorMean + posteriorSd * ext_rng_simulateStandardNormal(fit.control.rng);
  }
  
  double meanNormalGetPrediction(const BARTFit&, const Node& node, const double*, const double*)
  {
    DEFINE_SCRATCH(node);
    return scratch.mu;
  }
  
  void meanNormalGetPredictions(const BARTFit&, const Node& node, const double*, double* y_hat)
  {
    DEFINE_SCRATCH(node);
    y_hat[0] = scratch.mu;
  }
  
  void meanNormalGetPredictionsForIndices(const BARTFit&, const Node& node, const double*, const size_t*, double* y_hat)
  {
    DEFINE_SCRATCH(node);
    y_hat[0] = scratch.mu;
  }
  
  void meanNormalCopyScratch(const BARTFit&, Node& targetNode, const Node& sourceNode)
  {
    MeanNormalNodeScratch& target(*static_cast<MeanNormalNodeScratch*>(targetNode.getScratch()));
    const MeanNormalNodeScratch& source(*static_cast<const MeanNormalNodeScratch*>(sourceNode.getScratch()));
    
    target.mu = source.mu;
    target.numEffectiveObservations = source.numEffectiveObservations;
  }
  
  void meanNormalPrintScratch(const BARTFit&, const Node& node)
  {
    DEFINE_SCRATCH(node);
    ext_printf(" ave: %f", scratch.mu);
  }
  
  void meanNormalUpdateScratchWithValues(const BARTFit& fit, const Node& node, const double* y)
  {
    DEFINE_SCRATCH(node);
    
    size_t numObservations = node.getNumObservations();
    uint32_t updateType = (node.isTop() == false ? 1 : 0) + (fit.data.weights != NULL ? 2 : 0);
    switch (updateType) {
      case 0: // isTop && weights == NULL
      scratch.mu = ext_mt_computeMean(fit.threadManager, y, numObservations);
      scratch.numEffectiveObservations = static_cast<double>(numObservations);
      break;
      
      case 1: // !isTop && weights == NULL
      scratch.mu = ext_mt_computeIndexedMean(fit.threadManager, y, node.getObservationIndices(), numObservations);
      scratch.numEffectiveObservations = static_cast<double>(numObservations);
      break;
      
      case 2: // isTop && weights != NULL
      scratch.mu = ext_mt_computeWeightedMean(fit.threadManager, y, numObservations, fit.data.weights, &scratch.numEffectiveObservations);
      break;
      
      case 3: // !isTop && weights != NULL
      scratch.mu = ext_mt_computeIndexedWeightedMean(fit.threadManager, y, node.getObservationIndices(), numObservations, fit.data.weights, &scratch.numEffectiveObservations);
      break;
      
      default:
      break;
    }
  }
  
  void meanNormalUpdateScratchWithMemberships(const BARTFit&, const Node&) {

  }
  
  void meanNormalUpdateScratchFromChildren(const BARTFit&, Node& parentNode, const double*, const Node& leftChildNode, const Node& rightChildNode)
  {
    MeanNormalNodeScratch& parent(*static_cast<MeanNormalNodeScratch*>(parentNode.getScratch()));
    const MeanNormalNodeScratch& leftChild(*static_cast<const MeanNormalNodeScratch*>(leftChildNode.getScratch()));
    const MeanNormalNodeScratch& rightChild(*static_cast<const MeanNormalNodeScratch*>(rightChildNode.getScratch()));

    parent.numEffectiveObservations = leftChild.numEffectiveObservations + rightChild.numEffectiveObservations;    
    parent.mu = leftChild.mu * (leftChild.numEffectiveObservations / parent.numEffectiveObservations) +
                rightChild.mu * (rightChild.numEffectiveObservations / parent.numEffectiveObservations);
  }
  
  int meanNormalWriteScratch(const Node& node, ext_binaryIO* bio) {
    DEFINE_SCRATCH(node);
    
    int errorCode = ext_bio_writeDouble(bio, scratch.mu);
    if (errorCode != 0) return errorCode;
    
    return ext_bio_writeDouble(bio, scratch.numEffectiveObservations);
  }
  
  int meanNormalReadScratch(Node& node, ext_binaryIO* bio) {
    DEFINE_SCRATCH(node);
    
    int errorCode = ext_bio_readDouble(bio, &scratch.mu);
    if (errorCode != 0) return errorCode;
    
    return ext_bio_readDouble(bio, &scratch.numEffectiveObservations);
  }
  
  void meanNormalStoreScratch(const Node& node, void* targetPtr) {
    const MeanNormalNodeScratch& source(*static_cast<const MeanNormalNodeScratch*>(node.getScratch()));
    MeanNormalNodeScratch& target(*static_cast<MeanNormalNodeScratch*>(targetPtr));
    
    target.mu = source.mu;
    target.numEffectiveObservations = source.numEffectiveObservations;
  }
  
  void meanNormalRestoreScratch(Node& node, const void* sourcePtr) {
    const MeanNormalNodeScratch& source(*static_cast<const MeanNormalNodeScratch*>(sourcePtr));
    MeanNormalNodeScratch& target(*static_cast<MeanNormalNodeScratch*>(node.getScratch()));
    
    target.mu = source.mu;
    target.numEffectiveObservations = source.numEffectiveObservations;
  }

#undef DEFINE_SCRATCH
#undef DEFINE_MODEL

}

// ordinary linear regression with gaussian prior
namespace {
  using namespace dbarts;
  
  struct LinearRegressionNormalNodeScratch {
    double* posteriorCovarianceRightFactor;
    double* coefficients;
    double* Xt;
    double exponentialTerm;
  };
  
  double linearRegressionNormalLogIntegratedLikelihood(const BARTFit& fit, const Node& node, const double* y, double residualVariance);
  void linearRegressionNormalDrawFromPosterior(const BARTFit& fit, const Node& node, const double* y, double residualVariance);

  
  double linearRegressionNormalGetPrediction(const BARTFit& fit, const Node& node, const double*, const double* Xt);
  void linearRegressionNormalGetPredictions(const BARTFit& fit, const Node& node, const double*, double* y_hat);
  void linearRegressionNormalGetPredictionsForIndices(const BARTFit& fit, const Node& node, const double*, const std::size_t*, double* y_hat);
    
  void linearRegressionNormalCreateScratch(const BARTFit& fit, Node& node);
  void linearRegressionNormalDestroyScratch(const BARTFit& fit, Node& node);
  void linearRegressionNormalCopyScratch(const BARTFit& fit, Node& target, const Node& source);
  void linearRegressionNormalPrintScratch(const BARTFit& fit, const Node& node);
  
  void linearRegressionNormalUpdateScratchWithMemberships(const BARTFit& fit, const Node& node);
  void linearRegressionNormalUpdateScratchWithValues(const BARTFit& fit, const Node& node, const double* y);
  void linearRegressionNormalUpdateScratchWithMembershipsAndValues(const BARTFit& fit, const Node& node, const double* y);
  void linearRegressionNormalUpdateScratchFromChildren(const BARTFit& fit, Node& parent, const double* y, const Node& leftChild, const Node& rightChild);
/* 
  int linearRegressionNormalWriteScratch(const Node& node, ext_binaryIO* bio);
  int linearRegressionNormalReadScratch(Node& node, ext_binaryIO* bio);
  
  void linearRegressionNormalStoreScratch(const Node& node, void* target);
  void linearRegressionNormalRestoreScratch(Node& node, const void* source); */
}

namespace dbarts {
  namespace EndNode {
    LinearRegressionNormalModel* createLinearRegressionNormalModel(const BARTFit& fit)
    {
      LinearRegressionNormalModel* result = new LinearRegressionNormalModel;
    
      initializeLinearRegressionNormalModel(fit, *result);
    
      return result;
    }
  
    void initializeLinearRegressionNormalModel(const BARTFit& fit, LinearRegressionNormalModel& model)
    {
      model.info = CONDITIONALLY_INTEGRABLE;
      model.perNodeScratchSize = sizeof(LinearRegressionNormalNodeScratch);
      model.precisions = NULL;
      
      model.computeLogPrior = NULL;
      model.computeLogIntegratedLikelihood = &linearRegressionNormalLogIntegratedLikelihood;
      model.drawFromPrior = NULL;
      model.drawFromPosterior = &linearRegressionNormalDrawFromPosterior;
      
      model.getPrediction = &linearRegressionNormalGetPrediction;
      model.getPredictions = &linearRegressionNormalGetPredictions;
      model.getPredictionsForIndices = &linearRegressionNormalGetPredictionsForIndices;
      
      model.createScratch = &linearRegressionNormalCreateScratch;
      model.destroyScratch = &linearRegressionNormalDestroyScratch;
      model.copyScratch = &linearRegressionNormalCopyScratch;
      model.printScratch = &linearRegressionNormalPrintScratch;
      
      model.updateScratchWithValues = &linearRegressionNormalUpdateScratchWithValues;
      model.updateScratchWithMemberships = &linearRegressionNormalUpdateScratchWithMemberships;
      model.updateScratchWithMembershipsAndValues = &linearRegressionNormalUpdateScratchWithMembershipsAndValues;
      model.updateScratchFromChildren = &linearRegressionNormalUpdateScratchFromChildren;
      
      model.writeScratch = NULL; // &linearRegressionNormalWriteScratch;
      model.readScratch = NULL; // &linearRegressionNormalReadScratch;
      model.storeScratch = NULL; // &linearRegressionNormalStoreScratch;
      model.restoreScratch = NULL; // &linearRegressionNormalRestoreScratch;
      
      size_t numPredictors = fit.data.numPredictors + 1;
      model.Xt = new double[numPredictors * fit.data.numObservations];
      double* Xt = const_cast<double*>(model.Xt);
      
      for (size_t col = 0; col < fit.data.numObservations; ++col) {
        Xt[col * numPredictors] = 1.0;
        
        std::memcpy(Xt + col * numPredictors + 1,
                    fit.scratch.Xt + col * fit.data.numPredictors,
                    fit.data.numPredictors * sizeof(double));
      }
    }
    
    void destroyLinearRegressionNormalModel(LinearRegressionNormalModel* model)
    {
      invalidateLinearRegressionNormalModel(*model);
      
      delete model;
    }
    
    void invalidateLinearRegressionNormalModel(LinearRegressionNormalModel& model)
    {
      delete [] model.Xt; model.Xt = NULL;
    }
  }
}

namespace {

#define DEFINE_MODEL(_FIT_) const LinearRegressionNormalModel& model(*static_cast<const LinearRegressionNormalModel*>(_FIT_.model.endNodeModel));
#define DEFINE_SCRATCH(_NODE_) LinearRegressionNormalNodeScratch& scratch(*static_cast<LinearRegressionNormalNodeScratch*>(_NODE_.getScratch()));
  
  using dbarts::BARTFit;
  using dbarts::Node;
  
  using dbarts::EndNode::LinearRegressionNormalModel;
  
  double* createXtForNode(const BARTFit& fit, const Node& node);
  void calculateCovarianceRightFactor(const BARTFit& fit, const Node& node, const double* Xt, double* R);
  
  double linearRegressionNormalLogIntegratedLikelihood(const BARTFit& fit, const Node& node, const double*, double)
  {
    DEFINE_SCRATCH(node);
    
    size_t numPredictors = fit.data.numPredictors + 1;
    
    double determinantTerm = 0.0;
    for (size_t i = 0; i < numPredictors; ++i) determinantTerm -= log(scratch.posteriorCovarianceRightFactor[i * (1 + numPredictors)]);
    
    return determinantTerm + scratch.exponentialTerm;
  }
  
  void linearRegressionNormalDrawFromPosterior(const BARTFit& fit, const Node& node, const double*, double residualVariance)
  {
    DEFINE_SCRATCH(node);
    
    double sigma = std::sqrt(residualVariance);
    
    size_t numPredictors = fit.data.numPredictors + 1;
    
    for (size_t i = 0; i < numPredictors; ++i) scratch.coefficients[i] += ext_rng_simulateStandardNormal(fit.control.rng) * sigma;
    
    // coefficients become parameters
    ext_solveTriangularSystemInPlace(const_cast<const double*>(scratch.posteriorCovarianceRightFactor), numPredictors,
                                     false, EXT_TRIANGLE_TYPE_UPPER, scratch.coefficients, 1);
  }
  
  double linearRegressionNormalGetPrediction(const BARTFit& fit, const Node& node, const double*, const double* Xt)
  {
    DEFINE_SCRATCH(node);
    
    // be sneaky about intercept
    return scratch.coefficients[0] + ext_dotProduct(Xt, fit.data.numPredictors, scratch.coefficients + 1);
  }
  
  void linearRegressionNormalGetPredictions(const BARTFit& fit, const Node& node, const double*, double* y_hat)
  {
    DEFINE_MODEL(fit);
    DEFINE_SCRATCH(node);
    
    ext_multiplyMatrixIntoVector(model.Xt, fit.data.numPredictors + 1, fit.data.numObservations, true,
                                 scratch.coefficients, y_hat);
  }
  
  void linearRegressionNormalGetPredictionsForIndices(const BARTFit& fit, const Node& node, const double*, const std::size_t*, double* y_hat)
  {
    DEFINE_SCRATCH(node);
    
    ext_multiplyMatrixIntoVector(scratch.Xt, fit.data.numPredictors + 1, node.getNumObservations(), true,
                                 scratch.coefficients, y_hat);
  }
  
  void linearRegressionNormalCreateScratch(const BARTFit& fit, Node& node)
  {
    DEFINE_MODEL(fit);
    DEFINE_SCRATCH(node);
    
    size_t numPredictors = fit.data.numPredictors + 1;
    
    scratch.exponentialTerm = 0.0;
    scratch.Xt = const_cast<double*>(model.Xt);
    scratch.coefficients = new double [numPredictors];
    ext_setVectorToConstant(scratch.coefficients, numPredictors, 0.0);
    scratch.posteriorCovarianceRightFactor = new double[numPredictors * numPredictors];
  }
  
  void linearRegressionNormalDestroyScratch(const BARTFit& fit, Node& node)
  {
    DEFINE_MODEL(fit);
    DEFINE_SCRATCH(node);
    
    delete [] scratch.coefficients;
    scratch.coefficients = NULL;
    
    delete [] scratch.posteriorCovarianceRightFactor;
    scratch.posteriorCovarianceRightFactor = NULL;
  
    if (scratch.Xt != model.Xt) {
      delete [] scratch.Xt;
      scratch.Xt = NULL;
    }
  }
  
  void linearRegressionNormalCopyScratch(const BARTFit& fit, Node& targetNode, const Node& sourceNode)
  {
    DEFINE_MODEL(fit);
    LinearRegressionNormalNodeScratch& target(*static_cast<LinearRegressionNormalNodeScratch*>(targetNode.getScratch()));
    const LinearRegressionNormalNodeScratch& source(*static_cast<const LinearRegressionNormalNodeScratch*>(sourceNode.getScratch()));
    
    size_t numPredictors = fit.data.numPredictors + 1;
    
    target.exponentialTerm = source.exponentialTerm;
    std::memcpy(target.coefficients, source.coefficients, numPredictors * sizeof(double));
    std::memcpy(target.posteriorCovarianceRightFactor, source.posteriorCovarianceRightFactor, numPredictors * numPredictors * sizeof(double));
    
    if (target.Xt != model.Xt) delete [] target.Xt;
    if (source.Xt != model.Xt) {
      size_t numObservations = sourceNode.getNumObservations();
      target.Xt = new double [numPredictors * numObservations];
      std::memcpy(target.Xt, source.Xt, numPredictors * numObservations * sizeof(double));
    } else {
      target.Xt = const_cast<double*>(model.Xt);
    }
  }
  
  void linearRegressionNormalPrintScratch(const BARTFit& fit, const Node& node)
  {
    DEFINE_SCRATCH(node);
    
    size_t numToPrint = fit.data.numPredictors < 4 ? fit.data.numPredictors + 1 : 5;
    for (size_t i = 0; i < numToPrint; ++i) ext_printf(" %f", scratch.coefficients[i]);
  }
  
  void linearRegressionNormalUpdateScratchWithValues(const BARTFit& fit, const Node& node, const double* y)
  {
    DEFINE_SCRATCH(node);
    
    size_t numObservations = node.getNumObservations();
    size_t numPredictors = fit.data.numPredictors + 1;
    
    const double* nodeY = node.isTop() ? y : node.subsetVector(y);
    
    double* beta_tilde = ext_stackAllocate(numPredictors, double);
    double* y_hat = ext_stackAllocate(numObservations, double);
    
    // R^-T X'
    ext_solveTriangularSystemInPlace(const_cast<const double*>(scratch.posteriorCovarianceRightFactor), numPredictors,
                                     true, EXT_TRIANGLE_TYPE_UPPER, scratch.Xt, numObservations);
    
        
    // R^-T X' y
    // we use this term in draw from posterior
    ext_multiplyMatrixIntoVector(scratch.Xt, numPredictors, numObservations, false, nodeY, scratch.coefficients);
   
    
    std::memcpy(beta_tilde, const_cast<const double*>(scratch.coefficients), numPredictors * sizeof(double));
    
    // R^-1 R^-T X' y
    ext_solveTriangularSystemInPlace(const_cast<const double*>(scratch.posteriorCovarianceRightFactor), numPredictors,
                                     false, EXT_TRIANGLE_TYPE_UPPER, beta_tilde, 1);
    
    // X R^-1 R^-T X' y, aka y_hat(ish); ish b/c of prior means not projection
    ext_multiplyMatrixIntoVector(scratch.Xt, numPredictors, numObservations, true, const_cast<const double*>(beta_tilde), y_hat);
    
    // y_hat - y
    ext_addVectorsInPlace(nodeY, numObservations, -1.0, y_hat);
    
    scratch.exponentialTerm = 0.5 * ext_dotProduct(nodeY, numObservations, y_hat) / (fit.state.sigma * fit.state.sigma);
    
    ext_stackFree(y_hat);
    ext_stackFree(beta_tilde);
    
    if (nodeY != y) delete [] nodeY;
  }
  
  void linearRegressionNormalUpdateScratchWithMemberships(const BARTFit& fit, const Node& node) {
    DEFINE_MODEL(fit);
    DEFINE_SCRATCH(node);
    
    if (scratch.Xt != model.Xt) delete [] scratch.Xt;
        
    scratch.Xt = node.isTop() ? const_cast<double*>(model.Xt) : createXtForNode(fit, node);
    
    calculateCovarianceRightFactor(fit, node, scratch.Xt, scratch.posteriorCovarianceRightFactor);
  }
  
  void linearRegressionNormalUpdateScratchWithMembershipsAndValues(const BARTFit& fit, const Node& node, const double* y)
  {
    /* DEFINE_MODEL(fit);
    DEFINE_SCRATCH(node);
    
    if (scratch.Xt != model.Xt) delete [] scratch.Xt;
    
    scratch.Xt = node.isTop() ? const_cast<double*>(model.Xt) : createXtForNode(fit, node);
    
    calculateCovarianceRightFactor(fit, node, scratch.Xt, scratch.posteriorCovarianceRightFactor); */
    
    linearRegressionNormalUpdateScratchWithMemberships(fit, node);
    
    linearRegressionNormalUpdateScratchWithValues(fit, node, y);
  }
  
  void linearRegressionNormalUpdateScratchFromChildren(const BARTFit& fit, Node& parentNode, const double* y, const Node& leftChildNode, const Node& rightChildNode)
  {
    DEFINE_MODEL(fit);
    LinearRegressionNormalNodeScratch& parent(*static_cast<LinearRegressionNormalNodeScratch*>(parentNode.getScratch()));
    const LinearRegressionNormalNodeScratch& leftChild(*static_cast<const LinearRegressionNormalNodeScratch*>(leftChildNode.getScratch()));
    const LinearRegressionNormalNodeScratch& rightChild(*static_cast<const LinearRegressionNormalNodeScratch*>(rightChildNode.getScratch()));
    
    if (!parentNode.isTop()) {
      size_t numPredictors = fit.data.numPredictors + 1;
      parent.Xt = new double[numPredictors * (leftChildNode.getNumObservations() + rightChildNode.getNumObservations())];
      std::memcpy(parent.Xt, leftChild.Xt, numPredictors * leftChildNode.getNumObservations() * sizeof(double));
      std::memcpy(parent.Xt + numPredictors * leftChildNode.getNumObservations(), rightChild.Xt, numPredictors * rightChildNode.getNumObservations() * sizeof(double));
    } else {
      parent.Xt = const_cast<double*>(model.Xt);
    }
    
    calculateCovarianceRightFactor(fit, parentNode, parent.Xt, parent.posteriorCovarianceRightFactor);
    
    linearRegressionNormalUpdateScratchWithValues(fit, parentNode, y);
  }
  
  double* createXtForNode(const BARTFit& fit, const Node& node)
  {
    size_t numObservations = node.getNumObservations();
    size_t numPredictors = fit.data.numPredictors + 1;
    
    double* Xt = new double[numPredictors * numObservations];
    
    const size_t* observationIndices = node.getObservationIndices();
  
    for (size_t col = 0; col < numObservations; ++col) {
      Xt[col * numPredictors] = 1.0;
    
      std::memcpy(Xt + col * numPredictors + 1 /* skip first row */,
                 fit.scratch.Xt + observationIndices[col] * fit.data.numPredictors,
                 fit.data.numPredictors * sizeof(double));
    }
    
    return Xt;
  }
  
  void calculateCovarianceRightFactor(const BARTFit& fit, const Node& node, const double* Xt, double* R)
  {
    DEFINE_MODEL(fit);
    
    size_t numObservations = node.getNumObservations();
    size_t numPredictors = fit.data.numPredictors + 1;
        
    ext_getSingleMatrixCrossproduct(Xt, numPredictors, numObservations, R, true, EXT_TRIANGLE_TYPE_UPPER);
    
    // add in prior contribution
    for (size_t i = 0; i < numPredictors; ++i) R[i * (1 + numPredictors)] += model.precisions[i] * fit.state.sigma * fit.state.sigma;
    
    ext_getSymmetricPositiveDefiniteTriangularFactorizationInPlace(R, numPredictors, EXT_TRIANGLE_TYPE_UPPER);
  }
  
#undef DEFINE_SCRATCH
#undef DEFINE_MODEL

}
