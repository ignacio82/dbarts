#include "config.hpp"
#include <dbarts/responseModel.hpp>

#include <cmath> // sqrt
#include <cstring> // strncpy

#include <external/io.h>
#include <external/linearAlgebra.h>
#include <external/random.h>
#include <external/stats.h>
#include <external/stats_mt.h>

#include <dbarts/bartFit.hpp>
#include <dbarts/data.hpp>

using std::uint32_t;
using std::size_t;

namespace {
  namespace NormalChiSquared {
    using namespace dbarts;
    
    void print(const BARTFit& fit);
  
    void drawFromPosterior(const BARTFit& fit, const double* y, const double* totalFits);
  
    double getScale(const BARTFit& fit);
    void setScale(const BARTFit& fit, double scale);
    void updateWithWeights(const BARTFit& fit);
    
    const double* getParameters(const Response::Model& model);
    void setParameters(Response::Model& model, const double* parameters);
  }
}

namespace dbarts {
  namespace Response {
    NormalChiSquaredModel* createNormalChiSquaredModel(const Data& data, double degreesOfFreedom, double quantile)
    {
      NormalChiSquaredModel* result = new NormalChiSquaredModel;
      initializeNormalChiSquaredModel(*result, data, degreesOfFreedom, quantile);
      
      return result;
    }
    
    void initializeNormalChiSquaredModel(NormalChiSquaredModel& model, const Data& data, double degreesOfFreedom, double quantile)
    {
      model.info = HAS_SCALE_PARAMETER;
      strncpy(model.name, normalChiSquaredName, sizeof(model.name));
      model.numParameters = 1;
      
      model.degreesOfFreedom = degreesOfFreedom;
      model.numEffectiveObservations = data.weights == NULL ? static_cast<double>(data.numObservations) : ext_sumVectorElements(data.weights, data.numObservations);
      model.scale = ext_quantileOfChiSquared(1.0 - quantile, degreesOfFreedom) / degreesOfFreedom;
      
      model.print = &::NormalChiSquared::print;
      
      model.drawFromPosterior = &::NormalChiSquared::drawFromPosterior;
      
      model.getScale = &::NormalChiSquared::getScale;
      model.setScale = &::NormalChiSquared::setScale;
      
      model.updateWithWeights = &::NormalChiSquared::updateWithWeights;
      
      model.getParameters = &::NormalChiSquared::getParameters;
      model.setParameters = &::NormalChiSquared::setParameters;
    }
  }
}

namespace {

#define CONST_DEFINE_MODEL(_FIT_) const NormalChiSquaredModel& model(*static_cast<const NormalChiSquaredModel*>(_FIT_.model.responseModel))
#define DEFINE_MODEL(_FIT_) NormalChiSquaredModel& model(*const_cast<NormalChiSquaredModel*>(static_cast<NormalChiSquaredModel*>(fit.model.responseModel)));
  
  namespace NormalChiSquared {
    using dbarts::BARTFit;
    using dbarts::Response::NormalChiSquaredModel;
    
    void print(const BARTFit& fit) {
      CONST_DEFINE_MODEL(fit);
      
      double quantile = 1.0 - ext_percentileOfChiSquared(model.scale * model.degreesOfFreedom / (model.sigma * model.sigma), model.degreesOfFreedom);
      ext_printf("\tresponse - normal w/chisq; df = %f, quantile = %f\n",
                 model.degreesOfFreedom, quantile);
    }
    
    void drawFromPosterior(const BARTFit& fit, const double* y, const double* totalFits)
    {
      DEFINE_MODEL(fit);
      const Data& data(fit.data);
    
      double sumOfSquaredResiduals;
      if (data.weights != NULL) {
        sumOfSquaredResiduals = ext_mt_computeWeightedSumOfSquaredResiduals(fit.threadManager, y, data.numObservations, data.weights, totalFits);
      } else {
        sumOfSquaredResiduals = ext_mt_computeSumOfSquaredResiduals(fit.threadManager, y, data.numObservations, totalFits);
      }
    
      double posteriorDegreesOfFreedom = model.degreesOfFreedom + model.numEffectiveObservations;
      double posteriorScale = model.degreesOfFreedom * model.scale + sumOfSquaredResiduals;

      model.sigma = std::sqrt(posteriorScale / ext_rng_simulateChiSquared(fit.control.rng, posteriorDegreesOfFreedom));
    }
    
    double getScale(const BARTFit& fit) {
      CONST_DEFINE_MODEL(fit);
      return model.scale;
    }
    
    void setScale(const BARTFit& fit, double scale) {
      DEFINE_MODEL(fit);
      model.scale = scale;
    }
    
    void updateWithWeights(const BARTFit& fit) {
      DEFINE_MODEL(fit);

      model.numEffectiveObservations = fit.data.weights == NULL ? static_cast<double>(fit.data.numObservations) : ext_sumVectorElements(fit.data.weights, fit.data.numObservations);
    }
    
    const double* getParameters(const Response::Model& modelRef) {
      const NormalChiSquaredModel& model(static_cast<const NormalChiSquaredModel&>(modelRef));
      
      return &model.sigma;
    }
    
    void setParameters(Response::Model& modelRef, const double* parameters) {
      NormalChiSquaredModel& model(static_cast<NormalChiSquaredModel&>(modelRef));
      
      model.sigma = parameters[0];
    }
  } // namespace NormalChiSquared
} // anonymous namespace

#undef DEFINE_MODEL

