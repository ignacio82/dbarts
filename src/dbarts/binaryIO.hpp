#ifndef DBARTS_BINARY_IO_HPP
#define DBARTS_BINARY_IO_HPP

struct ext_binaryIO;

namespace dbarts {
  struct BARTFit;
  struct Control;
  struct Data;
  struct Model;
  
  bool writeControl(const Control& control, ext_binaryIO* bio);
  bool readControl(Control& control, ext_binaryIO* bio);

  bool writeData(const Data& data, ext_binaryIO* bio);
  bool readData(Data& data, ext_binaryIO* bio);
  
  bool writeModel(const Model& model, ext_binaryIO* bio);
  bool readModel(Model& model, ext_binaryIO* bio);
  
  bool writeState(const BARTFit& fit, ext_binaryIO* bio);
  bool readState(BARTFit& fit, ext_binaryIO* bio);
}

#endif // DBARTS_BINARY_IO_HPP
