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
//
// As to implementing these things: we use a bit of slight of hand to cache
// calculations. In general, the end node model should store the value of its
// parameters and be able to use those to make a prediction. The one exception to
// this is the following sequence of calls:
//   prepareScratchForLikelihoodAndPosteriorCalculations
//   computeLogIntegratedLikelihood
//   drawFromPosterior
// 
// The idea is that during prepareScratchForLikelihoodAndPosteriorCalculations,
// values used in *both* the likelihood and posterior steps can be cached. Until
// drawFromPosterior is called, the internal state, including the parameters, can be
// clobbered. Note that  the likelihood call may not be made on every node, so it does
// not make sense to precompute anything exclusive to that step.

namespace dbarts {
  struct BARTFit;
  struct Control;
  struct Data;
  
  struct Node;
  
  namespace EndNode {
    enum Options {
      NONE = 0x0,
      CONDITIONALLY_INTEGRABLE = 0x1, // should use log-integrated likelihood functions; otherwise log-prior
      PREDICTION_IS_CONSTANT   = 0x2, // uses dirty tricks to avoid a lot of unnecessary calls
      INVALID = 0x4
    };

    const char* const meanNormalName = "MnNl";
    const char* const linearRegressionNormalName = "LnRN";
    
    struct Model {
      std::size_t perNodeScratchSize;
      std::uint32_t info;
      char name[4];
      std::size_t numParameters;

      void (*print)(const BARTFit& fit);
      double* (*getParameters)(const BARTFit& fit, const Node& node);
      void (*setParameters)(const BARTFit& fit, Node& node, const double* parameters);
      
      double (*computeLogPrior)(const BARTFit& fit, const Node& node);
      double (*computeLogIntegratedLikelihood)(const BARTFit& fit, const Node& node, const double* y, double residualVariance);
      void (*drawFromPrior)(const BARTFit& fit, const Node& node);
      void (*drawFromPosterior)(const BARTFit& fit, const Node& node, const double* y, double residualVariance);
      
      double (*getPrediction)(const BARTFit& fit, const Node& node, const double* Xt); // at given Xt
      void (*getPredictions)(const BARTFit& fit, const Node& node, double* y_hat);     // at all of training, i.e. fit.scratch.Xt
      
      void (*createScratch)(const BARTFit& fit, Node& node);
      void (*destroyScratch)(const BARTFit& fit, void* scratch);
      void (*storeScratch)(const BARTFit& fit, const Node& source, void* target); // deep copies
      void (*restoreScratch)(const BARTFit& fit, void* source, Node& target); // destroys source as well
      
      void (*printScratch)(const BARTFit& fit, const Node &node);
      
      void (*updateScratchWithMemberships)(const BARTFit& fit, const Node& node, double residualVariance);
      void (*prepareScratchForLikelihoodAndPosteriorCalculations)(const BARTFit& fit, const Node& node, const double* y, double residualVariance);
      void (*updateMembershipsAndPrepareScratch)(const BARTFit& fit, const Node& node, const double* y, double residualVariance); // combines the above two
      
      // prepareScratchFromChildren - the parent is about to become the new end-node and can potentially scavenge some info from its children
      void (*prepareScratchFromChildren)(const BARTFit& fit, Node& parent, const double* y, double residualVariance, const Node& leftChild, const Node& rightChild);
      
      int (*writeScratch)(const Node& node, ext_binaryIO* bio);
      int (*readScratch)(Node& node, ext_binaryIO* bio);
    };
    
    struct MeanNormalModel : Model {
      double precision;
    };

#define DBARTS_END_NODE_MEAN_NORMAL_DEFAULT_K 2.0
    MeanNormalModel* createMeanNormalModel();
    void initializeMeanNormalModel(MeanNormalModel& model);
    MeanNormalModel* createMeanNormalModel(const Control& control, double k);
    void initializeMeanNormalModel(MeanNormalModel& model, const Control& control, double k);
    
    struct LinearRegressionNormalModel : Model {
      const double* Xt; // used to add a vector of ones for the constant term
      const double* precisions; // vector of inverse of prior variances
    };
  
    LinearRegressionNormalModel* createLinearRegressionNormalModel(const Data& data, const double* precisions);
    void destroyLinearRegressionNormalModel(LinearRegressionNormalModel* model);
    void initializeLinearRegressionNormalModel(LinearRegressionNormalModel& model, const Data& data, const double* precisions);
    void invalidateLinearRegressionNormalModel(LinearRegressionNormalModel& model);
  }
}

#endif // DBARTS_END_NODE_MODEL_HPP

