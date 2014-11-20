#include <R.h>
#include <Rdefines.h>

namespace dbarts {
  struct BARTFit;
  struct State;
  
  SEXP createStateExpressionFromFit(const BARTFit& fit); // sets PROTECT count to 1
  void initializeStateFromExpression(const BARTFit& fit, State& state, SEXP stateExpr);
  void storeStateExpressionFromFit(const BARTFit& fit, SEXP stateExpr);
}

