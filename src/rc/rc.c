#include "config.h"
#include <rc/rc.h>

#include <string.h>

#include <R.h>
#include <Rdefines.h>

SEXP rc_allocateInSlot(SEXP object, const char* name, SEXPTYPE type, size_t length)
{
  SEXP result = allocVector(type, (int) length);
  
  SET_SLOT(object, install(name), result);
  return result;
}

SEXP rc_setMatrixDimensions(SEXP object, size_t numRows, size_t numCols)
{
  SEXP dimsExpr = NEW_INTEGER(2);
  int* dims = INTEGER(dimsExpr);
  
  dims[0] = (int) numRows;
  dims[1] = (int) numCols;

  SET_ATTR(object, R_DimSymbol, dimsExpr);
  
  return object;
}

SEXP rc_setArrayDimensions(SEXP object, const size_t* dimensions, size_t numDimensions)
{
  SEXP dimsExpr = NEW_INTEGER((int) numDimensions);
  int* dims = INTEGER(dimsExpr);
  
  for (size_t i = 0; i < numDimensions; ++i) dims[i] = (int) dimensions[i];
  
  SET_ATTR(object, R_DimSymbol, dimsExpr);
  
  return object;
}

bool rc_isS4Null(SEXP object)
{
  if (!isSymbol(object)) return false;
  
  const char* symbolName = CHAR(PRINTNAME(object));
  if (strncmp(symbolName, "\1NULL\1", 6) == 0) return true;

  return false;
}

SEXP rc_duplicateVector(const double* x, size_t length)
{
  SEXP result = allocVector(REALSXP, (int) length);
  memcpy(REAL(result), x, length * sizeof(double));

  return result;
}

SEXP rc_duplicateMatrix(const double* x, size_t numRows, size_t numCols)
{
  SEXP result = allocVector(REALSXP, (int) (numRows * numCols));
  memcpy(REAL(result), x, numRows * numCols * sizeof(double));
  
  return rc_setMatrixDimensions(result, numRows, numCols);
}

SEXP rc_duplicateArray(const double* x, const size_t* dimensions, size_t numDimensions)
{
  size_t totalLength = 0;
  if (numDimensions > 0) {
    totalLength = dimensions[0];
    for (size_t i = 1; i < numDimensions; ++i) totalLength *= dimensions[i];
  }
  
  SEXP result = allocVector(REALSXP, (int) totalLength);
  memcpy(REAL(result), x, totalLength * sizeof(double));
  
  return rc_setArrayDimensions(result, dimensions, numDimensions);
}

void rc_copyFromVector(SEXP sourceExpr, double* target)
{
  memcpy(target, (const double*) REAL(sourceExpr), LENGTH(sourceExpr) * sizeof(double));
}

void rc_copyFromMatrix(SEXP sourceExpr, double* target)
{
  SEXP dimsExpr = GET_DIM(sourceExpr);
  int* dims = INTEGER(dimsExpr);
  
  memcpy(target, (const double*) REAL(sourceExpr), dims[0] * dims[1] * sizeof(double));
}

void rc_copyFromArray(SEXP sourceExpr, double* target)
{
  SEXP dimsExpr = GET_DIM(sourceExpr);
  size_t numDimensions = (size_t) LENGTH(dimsExpr);
  int* dims = INTEGER(dimsExpr);
  
  size_t totalLength = 0;
  if (numDimensions > 0) {
    totalLength = dims[0];
    for (size_t i = 1; i < numDimensions; ++i) totalLength *= dims[i];
  }

  memcpy(target, (const double*) REAL(sourceExpr), totalLength * sizeof(double));
}

bool rc_copyFromVectorAndCheckLength(SEXP sourceExpr, double* target, size_t length)
{
  if (length != (size_t) LENGTH(sourceExpr)) return false;
  
  memcpy(target, (const double*) REAL(sourceExpr), LENGTH(sourceExpr) * sizeof(double));
  return true;
}

int rc_copyFromMatrixAndCheckDimensions(SEXP sourceExpr, double* target, size_t numRows, size_t numCols)
{
  SEXP dimsExpr = GET_DIM(sourceExpr);
  if (LENGTH(dimsExpr) != 2) return -1;
  
  int* dims = INTEGER(dimsExpr);
  if (numRows != (size_t) dims[0]) return 1;
  if (numCols != (size_t) dims[1]) return 2;
  
  memcpy(target, (const double*) REAL(sourceExpr), numRows * numCols * sizeof(double));
  
  return 0;
}

int rc_copyFromArrayAndCheckDimensions(SEXP sourceExpr, double* target, const size_t* dimensions, size_t numDimensions)
{
  SEXP dimsExpr = GET_DIM(sourceExpr);
  if ((size_t) LENGTH(dimsExpr) != numDimensions) return -1;
  
  size_t totalLength = 0;
  if (numDimensions > 0) {
    int* dims = INTEGER(dimsExpr);
    if (dimensions[0] != (size_t) dims[0]) return 1;
    
    totalLength = dimensions[0];
    for (size_t i = 1; i < numDimensions; ++i) {
      if (dimensions[i] != (size_t) dims[i]) return i + 1;
      totalLength *= dimensions[i];
    }
  }
  
  memcpy(target, (const double*) REAL(sourceExpr), totalLength * sizeof(double));
  return 0;
}

void rc_copyIntoVector(const double* source, SEXP targetExpr)
{
  memcpy(REAL(targetExpr), source, LENGTH(targetExpr) * sizeof(double));
}

void rc_copyIntoMatrix(const double* source, SEXP targetExpr)
{
  int* dims = INTEGER(GET_DIM(targetExpr));
  memcpy(REAL(targetExpr), source, dims[0] * dims[1]);
}

void rc_copyIntoArray(const double* source, SEXP targetExpr)
{
  SEXP dimsExpr = GET_DIM(targetExpr);
  size_t numDimensions = LENGTH(targetExpr);
  int* dims = INTEGER(dimsExpr);
  
  size_t totalLength = 0;
  if (numDimensions > 0) {
    totalLength = dims[0];
    for (size_t i = 1; i < numDimensions; ++i) totalLength *= dims[i];
  }
  
  memcpy(REAL(targetExpr), source, totalLength * sizeof(double));
}

bool rc_copyIntoVectorAndCheckLength(const double* source, size_t length, SEXP targetExpr)
{
  if (length != (size_t) LENGTH(targetExpr)) return false;
  
  memcpy(REAL(targetExpr), source, length * sizeof(double));
  return true;
}

int rc_copyIntoMatrixAndCheckDimensions(const double* source, size_t numRows, size_t numCols, SEXP targetExpr)
{
  SEXP dimsExpr = GET_DIM(targetExpr);
  if (LENGTH(dimsExpr) != 2) return -1;
  
  int* dims = INTEGER(dimsExpr);
  if (numRows != (size_t) dims[0]) return 1;
  if (numCols != (size_t) dims[1]) return 2;

  memcpy(REAL(targetExpr), source, numRows * numCols * sizeof(double));
  return 0;
}

int  rc_copyIntoArrayAndCheckDimensions(const double* source, const size_t* dimensions, size_t numDimensions, SEXP targetExpr)
{
  SEXP dimsExpr = GET_DIM(targetExpr);
  if ((size_t) LENGTH(dimsExpr) != numDimensions) return -1;
  
  size_t totalLength = 0;
  if (numDimensions > 0) {
    int* dims = INTEGER(dimsExpr);
    if (dimensions[0] != (size_t) dims[0]) return 1;
    
    totalLength = dimensions[0];
    for (size_t i = 1; i < numDimensions; ++i) {
      if (dimensions[i] != (size_t) dims[i]) return i + 1;
      totalLength *= dimensions[i];
    }
  }
  
  memcpy(REAL(targetExpr), source, totalLength * sizeof(double));
  return 0;
}

