#ifndef DBARTS_END_NODE_MODEL_HPP
#define DBARTS_END_NODE_MODEL_HPP

#include <cstddef>
#include "cstdint.hpp"

struct ext_binaryIO;

// This is just C code using PODs of function pointers.
// 
// What we really want here is a variant on static polymorphism, which is to say
// that for the lifetime of a BARTFit object, only one type of end node model is
// ever encountered. Using virtual functions and inheritence yields a lot of
// unnecessary vtable lookups. This can also be achieved in C++ using the Curiously
// Recursive Template Pattern (CRTP), but that would require that BARTFits be
// templated and that in turns means that pretty much all code needs to be available
// to other compliation units. That would then defeat the purpose of a nice, happy
// C/C++ library accessed using R's mechanism for pushing around function pointers.
// Maybe (probably) templates are better, but that's just how things are (for now).
//
// That being said, end nodes are required to internally store their state, i.e.
// the parameters of the model. This is done by a "scratch". This scratch gets appended
// to nodes, but is only "active" on end-nodes. When an end-node is created
// a call to create the scratch is issued. If that node gives birth, the scratch
// will be destroyed. This can bloat the size of a node, so if a lot of memory needs
// to be allocated just make the official scratch a pointer to something elsewhere.

namespace dbarts {
  struct BARTFit;
  struct Control;
  
  struct Node;
  
  namespace EndNode {
    enum Options {
      NONE = 0x0,
      CONDITIONALLY_INTEGRABLE = 0x1, // should use log-integrated likelihood functions; otherwise log-prior
      PREDICTION_IS_CONSTANT   = 0x2, // uses dirty tricks to avoid a lot of unnecessary calls
      INVALID = 0x2
    };
    
    struct Model {
      std::size_t perNodeScratchSize;
      std::uint32_t info;
      
      double (*computeLogPrior)(const BARTFit& fit, const Node& node);
      double (*computeLogIntegratedLikelihood)(const BARTFit& fit, const Node& node, const double* y, double residualVariance);
      void (*drawFromPrior)(const BARTFit& fit, const Node& node);
      void (*drawFromPosterior)(const BARTFit& fit, const Node& node, const double* y, double residualVariance);
      
      double (*getPrediction)(const BARTFit& fit, const Node& node, const double* y, const double* Xt); // at given Xt
      void (*getPredictions)(const BARTFit& fit, const Node& node, const double* y, double* y_hat);     // at all of training, i.e. fit.scratch.Xt
      void (*getPredictionsForIndices)(const BARTFit& fit, const Node& node, const double* y, const std::size_t* indices, double* y_hat);
      
      void (*createScratch)(const BARTFit& fit, Node& node);
      void (*destroyScratch)(const BARTFit& fit, Node& node);
      void (*copyScratch)(const BARTFit& fit, Node& target, const Node& source);
      void (*printScratch)(const BARTFit& fit, const Node &node);
      
      // updateScratchWithValues - just the values of y (or residuals, really) have changed
      // updateScratchWithMemberships - obs in node are different, so new indices, new covariates
      void (*updateScratchWithValues)(const BARTFit& fit, const Node& node, const double* y);
      void (*updateScratchWithMemberships)(const BARTFit& fit, const Node& node);
      void (*updateScratchWithMembershipsAndValues)(const BARTFit& fit, const Node& node, const double* y);
      // updateScratchFromChildren - the parent is about to become the new end-node and can potentially scavenge some info from its children
      void (*updateScratchFromChildren)(const BARTFit& fit, Node& parent, const double* y, const Node& leftChild, const Node& rightChild);
      
      int (*writeScratch)(const Node& node, ext_binaryIO* bio);
      int (*readScratch)(Node& node, ext_binaryIO* bio);

      void (*storeScratch)(const Node& node, void* target);
      void (*restoreScratch)(Node& node, const void* source);
    };
    
    struct MeanNormalModel : Model {
      double precision;
    };
    
    MeanNormalModel* createMeanNormalModel();
    void initializeMeanNormalModel(MeanNormalModel& model);
    MeanNormalModel* createMeanNormalModel(const Control& control, double k);
    void initializeMeanNormalModel(MeanNormalModel& model, const Control& control, double k);
    
    struct LinearRegressionNormalModel : Model {
      const double* Xt; // used to add a vector of ones for the constant term
      const double* precisions; // vector of inverse of prior variances
    };
  
    LinearRegressionNormalModel* createLinearRegressionNormalModel(const BARTFit& fit);
    void destroyLinearRegressionNormalModel(LinearRegressionNormalModel* model);
    void initializeLinearRegressionNormalModel(const BARTFit& fit, LinearRegressionNormalModel& model);
    void invalidateLinearRegressionNormalModel(LinearRegressionNormalModel& model);
  }
}

#endif // DBARTS_END_NODE_MODEL_HPP
