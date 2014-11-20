#ifndef DBARTS_RESPONSE_MODEL_HPP
#define DBARTS_RESPONSE_MODEL_HPP

#include <cstddef>
#include "cstdint.hpp"

namespace dbarts {
  struct BARTFit;
  struct Data;
  struct Model;

  namespace Response {
    enum Options {
      NONE = 0x0,
      HAS_SCALE_PARAMETER = 0x1,
      INVALID = 0x2
    };
    
    const char* const normalChiSquaredName = "NlCS";
    
    struct Model {
      std::uint32_t info;
      char name[4];
      std::size_t numParameters;
      
      void (*print)(const BARTFit& fit);
      
      void (*drawFromPosterior)(const BARTFit& fit, const double* y, const double* totalFits);
      
      double (*getScale)(const BARTFit& fit);
      void (*setScale)(const BARTFit& fit, double scale);
      
      void (*updateWithWeights)(const BARTFit& fit);
      
      const double* (*getParameters)(const Response::Model& model);
      void (*setParameters)(Response::Model& model, const double* parameters);
    };
    
    struct NormalChiSquaredModel : Model {
      double degreesOfFreedom;
      double scale;
      
      double numEffectiveObservations;
      
      double sigma;
    };
    
#define DBARTS_RESPONSE_NORMAL_CHISQ_DEFAULT_DF       3.0
#define DBARTS_RESPONSE_NORMAL_CHISQ_DEFAULT_QUANTILE 0.9

    NormalChiSquaredModel* createNormalChiSquaredModel(const Data& data, double degreesOfFreedom, double quantile);
    void initializeNormalChiSquaredModel(NormalChiSquaredModel& model, const Data& data, double degreesOfFreedom, double quantile);
  } // namespace Response
} // namespace dbarts

#endif // DBARTS_RESPONSE_MODEL_HPP

