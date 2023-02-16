#include <iostream>
#include "pitts_parallel.hpp"
#include "pitts_common.hpp"
#include "pitts_tensortrain.hpp"
#include "pitts_tensortrain_axpby.hpp"
#include "pitts_tensortrain_random.hpp"
#include "pitts_eigen.hpp"
#include "pitts_tensor2_eigen_adaptor.hpp"


int main(int argc, char* argv[])
{
  PITTS::initialize(&argc, &argv);

  using Type = double;
  PITTS::TensorTrain<Type> TT1(10,100), TT2(10,100);
  TT2.setTTranks(150);
  TT1.setTTranks(20);
  randomize(TT1);
  randomize(TT2);
  Type tmp = 0;
  for(int iter = 0; iter < 10; iter++)
  {
    tmp += axpby(0.01, TT1, 0.9, TT2);
  }
  std::cout << "random: " << tmp << std::endl;

  PITTS::finalize();

  return 0;
}
