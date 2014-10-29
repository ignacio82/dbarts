#ifndef DBARTS_END_NODE_MODEL_HPP
#define DBARTS_END_NODE_MODEL_HPP

#include <cstddef>

struct ext_rng;

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
      double (*drawFromPosterior)(const Model& model, ext_rng* rng, double ybar, double numEffectiveObservations, double residualVariance);
      
      void (*clearScratch)(Node& node);
      void (*copyScratch)(Node& target, const Node& source);
      void (*printScratch)(const Node &node);
      
      void (*updateScratchWithResiduals)(const BARTFit& fit, const Node& node, const double* r);
      
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
