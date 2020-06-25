#include <mpi.h>
#include <omp.h>
#include <iostream>
#include "pitts_tensortrain.hpp"
#include "pitts_tensortrain_normalize.hpp"
#include "pitts_tensortrain_random.hpp"
#include "pitts_common.hpp"


int main(int argc, char* argv[])
{
  PITTS::initialize(&argc, &argv);

  using Type = double;
  PITTS::TensorTrain<Type> TT1(10,100);
  const int r = 20;
  TT1.setTTranks({r,r,r,r,r,r,r,r,r});
  randomize(TT1);
  Type tmp = 0;
  for(int iter = 0; iter < 1000; iter++)
  {
    tmp += normalize(TT1);
  }
  std::cout << "random: " << tmp << std::endl;

  PITTS::finalize();

  return 0;
}
