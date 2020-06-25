#include <mpi.h>
#include <omp.h>
#include <iostream>
#include "pitts_tensortrain.hpp"
#include "pitts_tensortrain_axpby.hpp"
#include "pitts_tensortrain_random.hpp"
#include "pitts_common.hpp"


int main(int argc, char* argv[])
{
  PITTS::initialize(&argc, &argv);

  using Type = double;
  PITTS::TensorTrain<Type> TT1(10,100), TT2(10,100);
  const int r = 10;
  TT2.setTTranks({r,r,r,r,r,r,r,r,r});
  randomize(TT2);
  TT1.setOnes();
  Type tmp = 0;
  for(int iter = 0; iter < 1000; iter++)
  {
    tmp += axpby(0.00001, TT1, 0.9, TT2);
  }
  std::cout << "random: " << tmp << std::endl;

  PITTS::finalize();

  return 0;
}
