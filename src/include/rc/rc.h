#ifndef RC_RC_H
#define RC_RC_H

#include <stdbool.h>
#include <stddef.h>

#include <Rinternals.h> // SEXP, SEXPTYPE

#ifdef __cplusplus
extern "C" {
#endif

SEXP rc_allocateInSlot(SEXP object, const char* name, SEXPTYPE type, size_t length);
SEXP rc_setMatrixDimensions(SEXP object, size_t numRows, size_t numCols);
SEXP rc_setArrayDimensions(SEXP object, const size_t* dimensions, size_t numDimensions);
bool rc_isS4Null(SEXP object);

SEXP rc_duplicateVector(const double* x, size_t length);
SEXP rc_duplicateMatrix(const double* x, size_t numRows, size_t numCols);
SEXP rc_duplicateArray(const double* x, const size_t* dimensions, size_t numDimensions);

void rc_copyFromVector(SEXP source, double* target);
void rc_copyFromMatrix(SEXP source, double* target);
void rc_copyFromArray(SEXP source, double* target);
bool rc_copyFromVectorAndCheckLength(SEXP source, double* target, size_t length);
// rc_copyFromMatrixAndCheckDimensions results:
//   0 - no failure
//  -1 - wrong number of dimensions in source
//   1 - numRows wrong
//   2 - numCols wrong
int rc_copyFromMatrixAndCheckDimensions(SEXP source, double* target, size_t numRows, size_t numCols);

// rc_copyFromArrayAndCheckDimensions results:
//   0 - no failure
//  -1 - wrong number of dimensions in source
//   n - dimension n does not match
int rc_copyFromArrayAndCheckDimensions(SEXP source, double* target, const size_t* dimensions, size_t numDimensions);

void rc_copyIntoVector(const double* source, SEXP target);
void rc_copyIntoMatrix(const double* source, SEXP target);
void rc_copyIntoArray(const double* source, SEXP target);
bool rc_copyIntoVectorAndCheckLength(const double* source, size_t length, SEXP target);
int  rc_copyIntoMatrixAndCheckDimensions(const double* source, size_t numRows, size_t numCols, SEXP target);
int  rc_copyIntoArrayAndCheckDimensions(const double* source, const size_t* dimensions, size_t numDimensions, SEXP target);

#ifdef __cplusplus
}
#endif

#endif // RC_RC_H

