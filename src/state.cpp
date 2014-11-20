#include "config.hpp"

#include <cstdlib> // free

#include <R.h>
#include <Rdefines.h>
#include <Rinternals.h>

#include <external/alloca.h>

#include <rc/rc.h>

#include <dbarts/bartFit.hpp>
#include <dbarts/control.hpp>
#include <dbarts/data.hpp>
#include <dbarts/endNodeModel.hpp>
#include <dbarts/model.hpp>
#include <dbarts/responseModel.hpp>
#include <dbarts/state.hpp>

namespace dbarts {
  SEXP createStateExpressionFromFit(const BARTFit& fit) {
    const Control& control(fit.control);
    const Data& data(fit.data);
    const Model& model(fit.model);
    const State& state(fit.state);
    
    SEXP result = PROTECT(NEW_OBJECT(MAKE_CLASS("dbartsState")));

    SET_SLOT(result, install("fit.tree"), rc_duplicateMatrix(state.treeFits, data.numObservations, control.numTrees));
    SET_SLOT(result, install("fit.total"), rc_duplicateVector(state.totalFits, data.numObservations));
    if (data.numTestObservations == 0) {
      SET_SLOT(result, install("fit.test"), NULL_USER_OBJECT);
    } else {
      SET_SLOT(result, install("fit.test"), rc_duplicateVector(state.totalTestFits, data.numTestObservations));
    }

    if (model.responseModel->numParameters > 0) {
      SET_SLOT(result, install("response.params"), rc_duplicateVector(model.responseModel->getParameters(*model.responseModel), model.responseModel->numParameters));
    }
    
    SET_SLOT(result, install("runningTime"), ScalarReal(state.runningTime));
    
    SEXP treeStructureExpr = rc_allocateInSlot(result, "tree.structure", STRSXP, control.numTrees);
    
    char** treeStrings = state.createTreeStructuresStrings(fit);
    for (size_t i = 0 ; i < control.numTrees; ++i) {
      SET_STRING_ELT(treeStructureExpr, static_cast<int>(i), CREATE_STRING_VECTOR(treeStrings[i]));
      std::free(treeStrings[i]);
    }
    delete [] treeStrings;
    
    if ((model.endNodeModel->info & EndNode::PREDICTION_IS_CONSTANT) != 0) {
      SEXP treeParameterExpr = rc_allocateInSlot(result, "tree.params", VECSXP, control.numTrees);
      size_t* treeParameterLengths;
      double** treeParameters = state.createTreeParametersVectors(fit, &treeParameterLengths);
      for (size_t i = 0; i < control.numTrees; ++i) {
        SET_VECTOR_ELT(treeParameterExpr, static_cast<int>(i), rc_duplicateVector(treeParameters[i], treeParameterLengths[i]));
        delete [] treeParameters[i];
      }
      delete [] treeParameters;
      delete [] treeParameterLengths;
    }
    
    return result;
  }
  
  void initializeStateFromExpression(const BARTFit& fit, State& state, SEXP stateExpr)
  {
    const Control& control(fit.control);
    const Data& data(fit.data);
    const Model& model(fit.model);
    
    rc_copyFromMatrix(GET_ATTR(stateExpr, install("fit.tree")), state.treeFits);
    rc_copyFromVector(GET_ATTR(stateExpr, install("fit.total")), state.totalFits);

    if (data.numTestObservations > 0)
      rc_copyFromVector(GET_ATTR(stateExpr, install("fit.test")), state.totalTestFits);
    
    // handle backwards compatibility
    SEXP responseParamExpr = GET_ATTR(stateExpr, install("sigma"));
    if (!isNull(responseParamExpr)) {
      if (strncmp(model.responseModel->name, Response::normalChiSquaredName, 4) != 0)
        error("attempting to use old-style state with new-style model that is not itself old");
      
      model.responseModel->setParameters(*model.responseModel, REAL(responseParamExpr));
    } else {
      if (model.responseModel->numParameters > 0)
        model.responseModel->setParameters(*model.responseModel, REAL(GET_ATTR(stateExpr, install("response.params"))));
    }
    
    state.runningTime = REAL(GET_ATTR(stateExpr, install("runningTime")))[0];
    
    // backwards compatibility
    SEXP treeStructureExpr = GET_ATTR(stateExpr, install("trees"));
    if (isNull(treeStructureExpr)) treeStructureExpr = GET_ATTR(stateExpr, install("tree.structure"));
    const char** treeStructureStrings = ext_stackAllocate(control.numTrees, const char*);
    for (size_t i = 0; i < control.numTrees; ++i)
      treeStructureStrings[i] = CHAR(STRING_ELT(treeStructureExpr, static_cast<int>(i)));
    state.recreateTreeStructuresFromStrings(fit, treeStructureStrings);
    ext_stackFree(treeStructureStrings);
    
    
    SEXP treeParameterExpr = GET_ATTR(stateExpr, install("tree.params"));
    if (!isNull(treeParameterExpr)) {
      double** treeParameters = ext_stackAllocate(control.numTrees, double*);
      for (size_t i = 0; i < control.numTrees; ++i)
        treeParameters[i] = REAL(VECTOR_ELT(treeParameterExpr, static_cast<int>(i)));
      state.setTreeParametersFromVectors(fit, treeParameters);
      ext_stackFree(treeParameters);
    } else {
      if ((model.endNodeModel->info & EndNode::PREDICTION_IS_CONSTANT) != 0)
        error("tree parameters are missing for end node model with non-constant predictions");
      state.setTreeParametersFromFits(fit);
    }
  }
  
  void storeStateExpressionFromFit(const BARTFit& fit, SEXP stateExpr)
  {
    const Control& control(fit.control);
    const Data& data(fit.data);
    const Model& model(fit.model);
    const State& state(fit.state);
    
    int errorCode = rc_copyIntoMatrixAndCheckDimensions(state.treeFits, data.numObservations, control.numTrees, GET_ATTR(stateExpr, install("fit.tree")));
    if (errorCode == -1) error("state@fit.tree must be a matrix");
    if (errorCode > 0) error("dimensions of state@fit.tree do not match sampler");
    
    if  (rc_copyIntoVectorAndCheckLength(state.totalFits, data.numObservations, GET_ATTR(stateExpr, install("fit.total"))) == false)
      error("length of state@fit.total does not match sampler");

    if (data.numTestObservations > 0 && rc_copyIntoVectorAndCheckLength(state.totalTestFits, data.numTestObservations, GET_ATTR(stateExpr, install("fit.test"))) == false)
      error("length of state@fit.test does not match sampler");
    
   
    // handle backwards compatibility
    SEXP responseParamExpr = GET_ATTR(stateExpr, install("sigma"));
    // TODO move sigma to the correct slot
    if (!isNull(responseParamExpr)) {
      if (strncmp(model.responseModel->name, Response::normalChiSquaredName, 4) != 0)
        error("attempting to use old-style state with new-style model that is not itself old");
      
      if (LENGTH(responseParamExpr) != 1)
        error("length of state@sigma is not 1");
      
      model.responseModel->setParameters(*model.responseModel, REAL(responseParamExpr));
    } else {
      responseParamExpr = GET_ATTR(stateExpr, install("response.params"));
      if (model.responseModel->numParameters > 0) {
        if (static_cast<size_t>(LENGTH(responseParamExpr)) != model.responseModel->numParameters)
          error("length of state@response.params does not match sampler");
        
        model.responseModel->setParameters(*model.responseModel, REAL(responseParamExpr));
      }
    }
    
    if (rc_copyIntoVectorAndCheckLength(&state.runningTime, 1, GET_ATTR(stateExpr, install("runningTime"))) == false)
      error("length of state@runningTime does not match sampler");
    

    // backwards compatibility
    SEXP treeStructureExpr = GET_ATTR(stateExpr, install("trees"));
    // TODO move trees to the correct slot
    if (isNull(treeStructureExpr)) treeStructureExpr = GET_ATTR(stateExpr, install("tree.structure"));
    if (static_cast<size_t>(LENGTH(treeStructureExpr)) != control.numTrees)
      error("length of state@tree.structure does not match sampler");
    
    char** treeStructureStrings = state.createTreeStructuresStrings(fit);
    for (size_t i = 0; i < control.numTrees; ++i) {
      SET_STRING_ELT(treeStructureExpr, static_cast<int>(i), CREATE_STRING_VECTOR(treeStructureStrings[i]));
      std::free(treeStructureStrings[i]);
    }
    delete [] treeStructureStrings;
    
    if ((model.endNodeModel->info & EndNode::PREDICTION_IS_CONSTANT) != 0) {
      SEXP treeParameterExpr = GET_ATTR(stateExpr, install("tree.params"));
      if (static_cast<size_t>(LENGTH(treeParameterExpr)) != control.numTrees)
        error("length of state@tree.params does not match sampler");
      
      size_t* treeParameterLengths;
      double** treeParameters = state.createTreeParametersVectors(fit, &treeParameterLengths);
      for (size_t i = 0; i < control.numTrees; ++i) {
        SET_VECTOR_ELT(treeParameterExpr, static_cast<int>(i), rc_duplicateVector(treeParameters[i], treeParameterLengths[i]));
        delete [] treeParameters[i];
      }
      delete [] treeParameters;
      delete [] treeParameterLengths;
    }
  }
}
