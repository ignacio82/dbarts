#ifndef DBARTS_END_NODE_MODEL_HPP
#define DBARTS_END_NODE_MODEL_HPP

#include <cstddef>

struct ext_rng;
struct ext_binaryIO;

namespace dbarts {
  struct BARTFit;
  struct Control;
  
  struct Node;
  
  namespace EndNode {
    enum Options {
      NONE = 0x0,
      CONDITIONALLY_INTEGRABLE = 0x1,
      INVALID = 0x2
    };
    
    struct Model {
      std::size_t perNodeScratchSize;
      Options info;
      
      double (*computeLogIntegratedLikelihood)(const Model& model, const BARTFit& fit, const Node& node, const double* y, double residualVariance);
      double (*drawFromPosterior)(const Model& model, const BARTFit& fit, const Node& node, double residualVariance);
      
      void (*clearScratch)(Node& node);
      void (*copyScratch)(Node& target, const Node& source);
      void (*printScratch)(const Node &node);
      
      // updateScratchWithValues - just the values of y (or residuals, really) have changed
      // updateScratchWithObservationsAndValues - the set of observations falling in the node changed as well, so our set of covariates is different
      void (*updateScratchWithValues)(const BARTFit& fit, const Node& node, const double* y);
      void (*updateScratchWithObservationsAndValues)(const BARTFit& fit, const Node& node, const double* y);
      // updateScratchFromChildren - the parent is about to become the new end-node and can potentially scavenge some info from its children
      void (*updateScratchFromChildren)(Node& parent, const Node& leftChild, const Node& rightChild);
      
      int (*writeScratch)(const Node& node, ext_binaryIO* bio);
      int (*readScratch)(Node& node, ext_binaryIO* bio);
      
      void (*storeScratch)(const Node& node, void* target);
      void (*restoreScratch)(Node& node, const void* source);
      
      virtual ~Model() { }
    };
    
    struct MeanNormalModel : Model {
      double precision;
      
      virtual ~MeanNormalModel() { }
    };
    
    MeanNormalModel* createMeanNormalModel();
    void initializeMeanNormalModel(MeanNormalModel& model);
    MeanNormalModel* createMeanNormalModel(const Control& control, double k);
    void initializeMeanNormalModel(MeanNormalModel& model, const Control& control, double k);
  }
}

#endif // DBARTS_END_NODE_MODEL_HPP
