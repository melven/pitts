#include "pitts_parallel.hpp"
#include "pitts_common.hpp"
#include "pitts_tensortrain.hpp"
#include "pitts_tensortrain_random.hpp"
#include "pitts_tensortrain_operator.hpp"
#include "pitts_tensortrain_operator_apply.hpp"
#include <iostream>


int main(int argc, char* argv[])
{
  const int d = 8;
  const int n = 100;
  const int r = 50;
  const int rOp = 3;

  PITTS::initialize(&argc, &argv);

  using Type = double;
  PITTS::TensorTrain<Type> TT1(d, n, r), TT2(d, n, r);
  PITTS::TensorTrainOperator<Type> TTOp(d, n, n, rOp);
  randomize(TT1);
  randomize(TT2);
  Type tmp = 0;

  for(int iter = 0; iter < 10; iter++)
  {
    apply(TTOp, TT1, TT2);
  }

  PITTS::finalize();

  return 0;
}
