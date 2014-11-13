#include "config.hpp"
#include <dbarts/endNodeModel.hpp>

#include <cmath>
#include "dbarts/cstdint.hpp"
#include <cstring>

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
    double average;
    double numEffectiveObservations;
  };
  
  double meanNormalLogIntegratedLikelihood(const BARTFit& fit, const Node& node, const double* y, double residualVariance);
  void meanNormalDrawFromPosterior(const BARTFit& fit, const Node& node, const double* y, double residualVariance);
  void meanNormalGetPredictions(const BARTFit& fit, const Node& node, const double* y, const double* x, double* y_hat);
  
  void meanNormalCreateScratch(const BARTFit& fit, Node& node);
  void meanNormalDeleteScratch(Node& node);
  void meanNormalCopyScratch(const BARTFit& fit, Node& target, const Node& source);
  void meanNormalPrintScratch(const BARTFit& fit, const Node& node);
  
  void meanNormalUpdateScratchWithValues(const BARTFit& fit, const Node& node, const double* y);
  void meanNormalUpdateScratchWithMemberships(const BARTFit& fit, const Node& node);
  void meanNormalUpdateScratchFromChildren(Node& parentNode, const Node& leftChildNode, const Node& rightChildNode);
  
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
      
      model.computeLogIntegratedLikelihood = &meanNormalLogIntegratedLikelihood;
      model.drawFromPosterior = &meanNormalDrawFromPosterior;
      model.getPredictions = &meanNormalGetPredictions;
      model.createScratch = &meanNormalCreateScratch;
      model.deleteScratch = &meanNormalDeleteScratch;
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
  
  double meanNormalLogIntegratedLikelihood(const BARTFit& fit, const Node& node, const double* y, double residualVariance)
  {
    size_t numObservationsInNode = node.getNumObservations();
    if (numObservationsInNode == 0) return 0.0;
    
    const MeanNormalModel& model(*static_cast<const MeanNormalModel*>(fit.model.endNodeModel));
    MeanNormalNodeScratch& scratch(*static_cast<MeanNormalNodeScratch*>(node.getScratch()));
    
    double y_bar = scratch.average;
    double var_y = computeVarianceForNode(fit, node, y, y_bar);
      
    double dataPrecision = scratch.numEffectiveObservations / residualVariance;
    
    double result;
    result  = 0.5 * std::log(model.precision / (model.precision + dataPrecision));
    result -= 0.5 * (var_y / residualVariance) * static_cast<double>(numObservationsInNode - 1);
    result -= 0.5 * ((model.precision * y_bar) * (dataPrecision * y_bar)) / (model.precision + dataPrecision);
    
    return result;
  }
  
  void meanNormalDrawFromPosterior(const BARTFit& fit, const Node& node, const double*, double residualVariance) {
    const MeanNormalModel& model(*static_cast<const MeanNormalModel*>(fit.model.endNodeModel));
    MeanNormalNodeScratch& scratch(*static_cast<MeanNormalNodeScratch*>(node.getScratch()));
    
    double posteriorPrecision = scratch.numEffectiveObservations / residualVariance;
  
    double posteriorMean = posteriorPrecision * scratch.average / (model.precision + posteriorPrecision);
    double posteriorSd   = 1.0 / std::sqrt(model.precision + posteriorPrecision);
  
    scratch.average = posteriorMean + posteriorSd * ext_rng_simulateStandardNormal(fit.control.rng);
  }
  
  void meanNormalGetPredictions(const BARTFit&, const Node& node, const double*, const double*, double* y_hat)
  {
    const MeanNormalNodeScratch& scratch(*static_cast<const MeanNormalNodeScratch*>(node.getScratch()));
    *y_hat = scratch.average;
  }
  
  void meanNormalCreateScratch(const BARTFit&, Node&) {

  }
  
  void meanNormalDeleteScratch(Node&) {

  }
  
  void meanNormalCopyScratch(const BARTFit&, Node& targetNode, const Node& sourceNode)
  {
    MeanNormalNodeScratch& target(*static_cast<MeanNormalNodeScratch*>(targetNode.getScratch()));
    const MeanNormalNodeScratch& source(*static_cast<const MeanNormalNodeScratch*>(sourceNode.getScratch()));
    
    target.average = source.average;
    target.numEffectiveObservations = source.numEffectiveObservations;
  }
  
  void meanNormalPrintScratch(const BARTFit&, const Node& node)
  {
    MeanNormalNodeScratch& scratch(*static_cast<MeanNormalNodeScratch*>(node.getScratch()));
    ext_printf(" ave: %f", scratch.average);
  }
  
  void meanNormalUpdateScratchWithValues(const BARTFit& fit, const Node& node, const double* y)
  {
    MeanNormalNodeScratch& scratch(*static_cast<MeanNormalNodeScratch*>(node.getScratch()));
    
    size_t numObservations = node.getNumObservations();
    uint32_t updateType = (node.isTop() == false ? 1 : 0) + (fit.data.weights != NULL ? 2 : 0);
    switch (updateType) {
      case 0: // isTop && weights == NULL
      scratch.average = ext_mt_computeMean(fit.threadManager, y, numObservations);
      scratch.numEffectiveObservations = static_cast<double>(numObservations);
      break;
      
      case 1: // !isTop && weights == NULL
      scratch.average = ext_mt_computeIndexedMean(fit.threadManager, y, node.getObservationIndices(), numObservations);
      scratch.numEffectiveObservations = static_cast<double>(numObservations);
      break;
      
      case 2: // isTop && weights != NULL
      scratch.average = ext_mt_computeWeightedMean(fit.threadManager, y, numObservations, fit.data.weights, &scratch.numEffectiveObservations);
      break;
      
      case 3: // !isTop && weights != NULL
      scratch.average = ext_mt_computeIndexedWeightedMean(fit.threadManager, y, node.getObservationIndices(), numObservations, fit.data.weights, &scratch.numEffectiveObservations);
      break;
      
      default:
      break;
    }
  }
  
  void meanNormalUpdateScratchWithMemberships(const BARTFit&, const Node&) {

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

// ordinary linear regression with gaussian prior
namespace {
  using namespace dbarts;
  
  struct LinearRegressionNormalNodeScratch {
    double* Xt;
    double* posteriorCovarianceRightFactor;
    double* coefficients;
  };
  
  double linearRegressionNormalLogIntegratedLikelihood(const BARTFit& fit, const Node& node, const double* y, double residualVariance);
  void linearRegressionNormalDrawFromPosterior(const BARTFit& fit, const Node& node, const double* y, double residualVariance);

  void linearRegressionNormalCreateScratch(const BARTFit& fit, Node& node);
  void linearRegressionNormalDeleteScratch(Node& node);
  void linearRegressionNormalCopyScratch(const BARTFit& fit, Node& target, const Node& source);
  void linearRegressionNormalPrintScratch(const BARTFit& fit, const Node& node);
  
  void linearRegressionNormalUpdateScratchWithMemberships(const BARTFit& fit, const Node& node);
  void linearRegressionNormalUpdateScratchWithValues(const BARTFit& fit, const Node& node, const double* r);
/*  void linearRegressionNormalUpdateScratchFromChildren(Node& parentNode, const Node& leftChildNode, const Node& rightChildNode);
  
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
      model.perNodeScratchSize = sizeof(LinearRegressionNormalModel);
      model.precisions = NULL;
      
      model.computeLogIntegratedLikelihood = &linearRegressionNormalLogIntegratedLikelihood;
      model.drawFromPosterior = &linearRegressionNormalDrawFromPosterior;
      model.createScratch = &linearRegressionNormalCreateScratch;
      model.deleteScratch = &linearRegressionNormalDeleteScratch;
      model.copyScratch = &linearRegressionNormalCopyScratch;
      model.printScratch = &linearRegressionNormalPrintScratch;
      model.updateScratchWithValues = &linearRegressionNormalUpdateScratchWithValues;
      model.updateScratchWithMemberships = &linearRegressionNormalUpdateScratchWithMemberships;
      /* model.updateScratchWithMembershipsAndValues = &linearRegressionNormalUpdateScratchWithValues;
      model.updateScratchFromChildren = &linearRegressionNormalUpdateScratchFromChildren;
      model.writeScratch = &linearRegressionNormalWriteScratch;
      model.readScratch = &linearRegressionNormalReadScratch;
      model.storeScratch = &linearRegressionNormalStoreScratch;
      model.restoreScratch = &linearRegressionNormalRestoreScratch; */
      
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
  using dbarts::BARTFit;
  using dbarts::Node;
  
  using dbarts::EndNode::LinearRegressionNormalModel;
  
  double* createXtForNode(const BARTFit& fit, const Node& node);
  double* createCovarianceRightFactor(const BARTFit& fit, const Node& node, const double* Xt);
  
  void linearRegressionNormalDrawFromPosterior(const BARTFit& fit, const Node& node, const double* allY, double residualVariance)
  {
    const LinearRegressionNormalModel& model(*static_cast<const LinearRegressionNormalModel*>(fit.model.endNodeModel));
    LinearRegressionNormalNodeScratch& scratch(*static_cast<LinearRegressionNormalNodeScratch*>(node.getScratch()));
    
    double sigma = std::sqrt(residualVariance);
    
    size_t numPredictors = fit.data.numPredictors + 1;
    size_t numObservations = node.getNumObservations();
    
    double* Xt;
    if (!node.isTop()) Xt = createXtForNode(fit, node);
    else {
      Xt = new double[numPredictors * numObservations];
      std::memcpy(Xt, model.Xt, numPredictors * numObservations * sizeof(double));
    }
    double* R = createCovarianceRightFactor(fit, node, Xt);
    
    ext_solveTriangularSystemInPlace(const_cast<const double*>(R), numPredictors, true, EXT_TRIANGLE_TYPE_UPPER, Xt, numObservations);
    
    const double* y;
    if (node.isTop()) y = allY;
    else {
      y = new double[numObservations];
      const size_t* observationIndices = node.getObservationIndices();
      
      for (size_t i = 0; i < numObservations; ++i) const_cast<double*>(y)[i] = allY[observationIndices[i]];
    }
    
    ext_multiplyMatrixIntoVector(Xt, numPredictors, numObservations, false, y, scratch.coefficients);
    
    for (size_t i = 0; i < numPredictors; ++i) scratch.coefficients[i] += ext_rng_simulateStandardNormal(fit.control.rng) * sigma;
    
    ext_solveTriangularSystemInPlace(const_cast<const double*>(R), numPredictors, false, EXT_TRIANGLE_TYPE_UPPER, scratch.coefficients, 1);
    
    if (!node.isTop()) delete [] y;
    delete [] R;
    delete [] Xt;
  }
  
  double linearRegressionNormalLogIntegratedLikelihood(const BARTFit& fit, const Node& node, const double* allY, double residualVariance)
  {
    const LinearRegressionNormalModel& model(*static_cast<const LinearRegressionNormalModel*>(fit.model.endNodeModel));
    
    size_t numPredictors = fit.data.numPredictors + 1;
    size_t numObservations = node.getNumObservations();
    
    double* Xt;
    if (!node.isTop()) Xt = createXtForNode(fit, node);
    else {
      Xt = new double[numPredictors * numObservations];
      std::memcpy(Xt, model.Xt, numPredictors * numObservations * sizeof(double));
    }
    double* R = createCovarianceRightFactor(fit, node, Xt);
    

    
    double determinantTerm = 0.0;
    for (size_t i = 0; i < numPredictors; ++i) determinantTerm -= log(R[i * (1 + numPredictors)]);
    
    // R^-T Xt
    ext_solveTriangularSystemInPlace(const_cast<const double*>(R), numPredictors, true, EXT_TRIANGLE_TYPE_UPPER, Xt, numObservations);
    
    const double* y;
    if (node.isTop()) y = allY;
    else {
      y = new double[numObservations];
      const size_t* observationIndices = node.getObservationIndices();
      
      for (size_t i = 0; i < numObservations; ++i) const_cast<double*>(y)[i] = allY[observationIndices[i]];
    }
    
    double* beta_tilde = new double[numPredictors];
    double* y_hat = new double[numObservations];
    
    ext_multiplyMatrixIntoVector(Xt, numPredictors, numObservations, false, y, beta_tilde);
    ext_multiplyMatrixIntoVector(Xt, numPredictors, numObservations, true, beta_tilde, y_hat);
    
    // y_hat - y
    ext_addVectorsInPlace(y, numObservations, -1.0, y_hat);
    
    double exponentialTerm = 0.5 * ext_dotProduct(y, numObservations, y_hat) / residualVariance;
    
    delete [] y_hat;
    delete [] beta_tilde;
    if (!node.isTop()) delete [] y;
    delete [] R;
    delete [] Xt;
    
    return determinantTerm + exponentialTerm;
  }
  
  void linearRegressionNormalCreateScratch(const BARTFit& fit, Node& node)
  {
    LinearRegressionNormalNodeScratch& scratch(*static_cast<LinearRegressionNormalNodeScratch*>(node.getScratch()));
    
    size_t numPredictors = fit.data.numPredictors + 1;
    
    scratch.coefficients = new double [numPredictors];
    ext_setVectorToConstant(scratch.coefficients, numPredictors, 0.0);
    scratch.posteriorCovarianceRightFactor = new double[numPredictors * numPredictors];
  }
  
  void linearRegressionNormalDeleteScratch(Node& node)
  {
    LinearRegressionNormalNodeScratch& scratch(*static_cast<LinearRegressionNormalNodeScratch*>(node.getScratch()));
    
    delete [] scratch.posteriorCovarianceRightFactor;
    scratch.posteriorCovarianceRightFactor = NULL;
    
    delete [] scratch.coefficients;
    scratch.coefficients = NULL;
  }
  
  void linearRegressionNormalCopyScratch(const BARTFit& fit, Node& targetNode, const Node& sourceNode)
  {
    LinearRegressionNormalNodeScratch& target(*static_cast<LinearRegressionNormalNodeScratch*>(targetNode.getScratch()));
    const LinearRegressionNormalNodeScratch& source(*static_cast<const LinearRegressionNormalNodeScratch*>(sourceNode.getScratch()));
    
    size_t numPredictors = fit.data.numPredictors + 1;
    
    std::memcpy(target.coefficients, source.coefficients, numPredictors * sizeof(double));
    std::memcpy(target.posteriorCovarianceRightFactor, source.posteriorCovarianceRightFactor, numPredictors * numPredictors * sizeof(double));
  }
  
  void linearRegressionNormalPrintScratch(const BARTFit& fit, const Node& node)
  {
    LinearRegressionNormalNodeScratch& scratch(*static_cast<LinearRegressionNormalNodeScratch*>(node.getScratch()));
    
    size_t numToPrint = fit.data.numPredictors < 4 ? fit.data.numPredictors + 1 : 5;
    for (size_t i = 0; i < numToPrint; ++i) ext_printf(" %f", scratch.coefficients[i]);
  }
  
  void linearRegressionNormalUpdateScratchWithValues(const BARTFit&, const Node&, const double*)
  {
    
  }
  
  void linearRegressionNormalUpdateScratchWithMemberships(const BARTFit&, const Node&) {
    /* LinearRegressionNormalModel& model(*static_cast<LinearRegressionNormalModel*>(fit.model.endNodeModel));
    LinearRegressionNormalNodeScratch& scratch(*static_cast<LinearRegressionNormalNodeScratch*>(node.getScratch()));
    
    if (scratch.Xt != model.Xt) delete [] scratch.Xt;
    
    size_t numObservations = node.getNumObservations();
    size_t numPredictors = fit.data.numPredictors + 1;
    
    scratch.Xt = node.isTop() ? model.Xt : createXtForNode(fit, node);
    
    double* XtX = scratch.posteriorCovarianceRightFactor;
    ext_getSingleMatrixCrossproduct(scratch.Xt, numPredictors, numObservations, XtX, true, EXT_TRIANGLE_TYPE_UPPER);
    
    // add in prior contribution
    for (size_t i = 0; i < numPredictors; ++i) XtX[i * (1 + numPredictors)] += model.precisions[i] * fit.state.sigma * fit.state.sigma;
    
    ext_getSymmetricPositiveDefiniteTriangularFactorizationInPlace(scratch.posteriorCovarianceRightFactor, numPredictors, EXT_TRIANGLE_TYPE_UPPER); */
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
  
  double* createCovarianceRightFactor(const BARTFit& fit, const Node& node, const double* Xt)
  {
    const LinearRegressionNormalModel& model(*static_cast<const LinearRegressionNormalModel*>(fit.model.endNodeModel));
    // LinearRegressionNormalNodeScratch& scratch(*static_cast<LinearRegressionNormalNodeScratch*>(node.getScratch()));
    
    size_t numObservations = node.getNumObservations();
    size_t numPredictors = fit.data.numPredictors + 1;
        
    double* result = new double[numPredictors * numPredictors];
    ext_getSingleMatrixCrossproduct(Xt, numPredictors, numObservations, result, true, EXT_TRIANGLE_TYPE_UPPER);
    
    // add in prior contribution
    for (size_t i = 0; i < numPredictors; ++i) result[i * (1 + numPredictors)] += model.precisions[i] * fit.state.sigma * fit.state.sigma;
    
    ext_getSymmetricPositiveDefiniteTriangularFactorizationInPlace(result, numPredictors, EXT_TRIANGLE_TYPE_UPPER);
    
    return result;
  }
}
