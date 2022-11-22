#include "pitts_parallel.hpp"
#include "pitts_common.hpp"
#include "pitts_tensortrain.hpp"
#include "pitts_tensortrain_dot.hpp"
#include "pitts_tensortrain_random.hpp"
#include <iostream>


int main(int argc, char* argv[])
{
  PITTS::initialize(&argc, &argv);

  using Type = double;
  PITTS::TensorTrain<Type> TT1(4,100), TT2(4,100);
  TT1.setTTranks(150);
  TT2.setTTranks({150,150,17});
  randomize(TT1);
  randomize(TT2);
  Type tmp = 0;
  for(int iter = 0; iter < 100; iter++)
  {
    tmp += dot(TT1,TT2);
  }
  std::cout << "random: " << tmp << std::endl;

  PITTS::finalize();

  return 0;
}
