#include <mpi.h>
#include <omp.h>
#include <iostream>
#include "pitts_tensortrain.hpp"
#include "pitts_tensortrain_dot.hpp"
#include "pitts_tensortrain_random.hpp"
#include "pitts_common.hpp"


int main(int argc, char* argv[])
{
  PITTS::initialize(&argc, &argv);

  using Type = double;
  PITTS::TensorTrain<Type> TT1(10,100), TT2(10,100);
  const int r = 20;
  TT1.setTTranks({r,r,r,r,r,r,r,r,r});
  TT2.setTTranks({r,r,r,r,r,r,r,r,r});
  randomize(TT1);
  randomize(TT2);
  Type tmp = 0;
  for(int iter = 0; iter < 1000; iter++)
  {
    tmp += dot(TT1,TT2);
  }
  std::cout << "random: " << tmp << std::endl;

  PITTS::finalize();

  return 0;
}
