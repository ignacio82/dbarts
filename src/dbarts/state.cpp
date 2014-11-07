#include "config.hpp"
#include <dbarts/state.hpp>

#include <cerrno>
#include <cstdlib> // malloc/free
#include <cstring> // strerror

#include <external/io.h>
#include <external/stringWriter.h>

#include <dbarts/bartFit.hpp>
#include <dbarts/control.hpp>

#include "tree.hpp"

namespace dbarts {
  const char* const* State::createTreeStrings(const BARTFit& fit) const
  {
    ext_stringWriter writer;
    
    int errorCode = 0;
    
    char** result = new char*[fit.control.numTrees];
    
    size_t treeNum = 0;
    for ( ; treeNum < fit.control.numTrees; ++treeNum) {
      ext_swr_initialize(&writer, EXT_SWR_DEFAULT_BUFFER_SIZE);
      
      if ((errorCode = TREE_AT(trees, treeNum, fit.scratch.nodeSize)->write(&writer)) != 0) break;
      if ((errorCode = ext_swr_writeChar(&writer, '\0')) != 0) break;
      
      result[treeNum] = writer.buffer;
    }
    
    if (errorCode != 0) {
      for ( ; treeNum < fit.control.numTrees; --treeNum) std::free(result[treeNum]);
      delete [] result;
      
      ext_throwError("unable to write tree %lu: %s", treeNum + 1, std::strerror(errno));
    }
    
    return result;
  }
  
  void State::recreateTreesFromStrings(const BARTFit& fit, const char* const* treeStrings)
  {
    size_t treeNum = 0;
    
    int errorCode = 0;
    
    for (; treeNum < fit.control.numTrees; ++treeNum) {
      errorCode = TREE_AT(trees, treeNum, fit.scratch.nodeSize)->read(fit, treeStrings[treeNum]);
      
      if (errorCode != 0) break;
    }
    
    if (errorCode != 0) {
      for ( ; treeNum < fit.control.numTrees; --treeNum) TREE_AT(trees, treeNum, fit.scratch.nodeSize)->clear(fit);
      
      ext_throwError("unable to read tree %lu: %s", treeNum + 1, std::strerror(errno));
    }
  }
}
