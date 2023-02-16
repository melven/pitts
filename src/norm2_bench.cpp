#include <iostream>
#include "pitts_parallel.hpp"
#include "pitts_common.hpp"
#include "pitts_tensortrain.hpp"
#include "pitts_tensortrain_norm.hpp"
#include "pitts_tensortrain_random.hpp"
#include "pitts_tensor2_eigen_adaptor.hpp"


int main(int argc, char* argv[])
{
  PITTS::initialize(&argc, &argv);

  using Type = double;
  PITTS::TensorTrain<Type> TT1(10,10000);
  const int r = 20;
  TT1.setTTranks({r,r,r,r,r,r,r,r,r});
  randomize(TT1);
  Type tmp = 0;
  for(int iter = 0; iter < 100; iter++)
  {
    tmp += norm2(TT1);
  }
  std::cout << "random: " << tmp << std::endl;

  PITTS::finalize();

  return 0;
}
