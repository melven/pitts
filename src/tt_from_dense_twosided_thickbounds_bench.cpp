#include <charconv>
#include <vector>
#include <stdexcept>
#include <iostream>
#include "pitts_common.hpp"
#include "pitts_multivector.hpp"
#include "pitts_multivector_random.hpp"
#include "pitts_tensortrain.hpp"
#include "pitts_tensortrain_from_dense_twosided.hpp"
#include "pitts_tensor2_eigen_adaptor.hpp"
#include "pitts_multivector_eigen_adaptor.hpp"



int main(int argc, char* argv[])
{
  PITTS::initialize(&argc, &argv);

  if( argc != 5 )
    throw std::invalid_argument("Requires 4 arguments (n d max_r nIter)!");

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

  // compress shape, s.t. first and last dimensions are bigger than max_r
  const double nMin = std::max<double>(2*max_r, 10.);
  while( shape.size() > 2 && shape.front() < nMin )
  {
    shape[1] *= shape[0];
    shape.erase(shape.begin());
  }
  while( shape.size() > 2 && shape.back() < nMin )
  {
    n = shape.size();
    shape[n-2] *= shape[n-1];
    shape.pop_back();
  }

  PITTS::MultiVector<double> data(nTotal/shape.back(), shape.back());
  randomize(data);

  PITTS::MultiVector<double> X(nTotal/shape.back(), shape.back());
  PITTS::MultiVector<double> work(nTotal/shape.back(), shape.back());

  for(int iter = 0; iter < nIter; iter++)
  {
    copy(data, X);
    const auto TT = fromDense_twoSided(X, work, shape, 1.e-8, max_r);
  }

  PITTS::finalize();

  return 0;
}

