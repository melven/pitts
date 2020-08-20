#include "pitts_common.hpp"
#include "pitts_multivector.hpp"
#include "pitts_multivector_random.hpp"
#include "pitts_tensortrain.hpp"
#include "pitts_tensortrain_from_dense_classical.hpp"
#include <charconv>
#include <vector>



int main(int argc, char* argv[])
{
  PITTS::initialize(&argc, &argv);

  if( argc != 5 )
    throw std::invalid_argument("Requires 4 arguments!");

  std::size_t n = 0, d = 0, max_r = 0, nIter = 0;
  std::from_chars(argv[1], argv[2], n);
  std::from_chars(argv[2], argv[3], d);
  std::from_chars(argv[3], argv[4], max_r);
  std::from_chars(argv[4], argv[5], nIter);

  std::size_t nTotal = 1;
  std::vector<int> shape(d);
  for(int i = 0; i < d; i++)
  {
    nTotal *= n;
    shape[i] = n;
  }

  PITTS::MultiVector<double> data(nTotal, 1);
  randomize(data);

  for(int iter = 0; iter < nIter; iter++)
  {
    const auto TT = PITTS::fromDense_classical(&data(0,0), &data(0,0)+nTotal, shape, 1.e-8, max_r);
  }

  PITTS::finalize();

  return 0;
}

