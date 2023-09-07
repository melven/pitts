#include "pitts_parallel.hpp"
#include "pitts_common.hpp"
#include "pitts_tensortrain.hpp"
#include "pitts_tensortrain_random.hpp"
#include "pitts_tensortrain_operator.hpp"
#include "pitts_tensortrain_operator_apply_dense.hpp"
#include "pitts_multivector.hpp"
#include "pitts_multivector_random.hpp"
#include <iostream>
#include <charconv>


int main(int argc, char* argv[])
{
  if( argc != 5 )
    throw std::invalid_argument("Requires 4 arguments (n r rOp nIter)!");

  int n = 0, r = 0, rOp = 0, nIter = 0;
  std::from_chars(argv[1], argv[2], n);
  std::from_chars(argv[2], argv[3], r);
  std::from_chars(argv[3], argv[4], rOp);
  std::from_chars(argv[4], argv[5], nIter);

  const std::vector<int> dims = {r,n,r};
  const long long N = (long long)(r) * (long long)(n) * (long long)(r);

  PITTS::initialize(&argc, &argv);

  using Type = double;
  PITTS::TensorTrainOperator<Type> TTOp(dims, dims, rOp);
  randomize(TTOp.tensorTrain());
  PITTS::MultiVector<Type> mvX(N,1), mvY(N,1);
  randomize(mvX);
  randomize(mvY);

  for(int iter = 0; iter < nIter; iter++)
    apply(TTOp, mvX, mvY);
  
  // special padded / faster variant
  PITTS::TTOpApplyDenseHelper TTOpHelper(TTOp);
  TTOpHelper.addPadding(mvX);
  TTOpHelper.addPadding(mvY);

  for(int iter = 0; iter < nIter; iter++)
    apply(TTOpHelper, mvX, mvY);
  
  TTOpHelper.removePadding(mvX);
  TTOpHelper.removePadding(mvY);
  

  PITTS::finalize();

  return 0;
}
