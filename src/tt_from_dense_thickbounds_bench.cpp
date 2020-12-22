#include "pitts_parallel.hpp"
#include "pitts_common.hpp"
#include "pitts_multivector.hpp"
#include "pitts_multivector_random.hpp"
#include "pitts_tensortrain.hpp"
#include "pitts_tensortrain_from_dense.hpp"
#include <charconv>
#include <vector>



int main(int argc, char* argv[])
{
  PITTS::initialize(&argc, &argv);

  if( argc != 5 && argc != 6 )
    throw std::invalid_argument("Requires 4 or 5 arguments (n d max_r nIter [f_bound])!");

  std::size_t n = 0, d = 0, max_r = 0, nIter = 0;
  std::from_chars(argv[1], argv[2], n);
  std::from_chars(argv[2], argv[3], d);
  std::from_chars(argv[3], argv[4], max_r);
  std::from_chars(argv[4], argv[5], nIter);
  std::size_t f_bound = 2;
  if( argc == 6 )
    std::from_chars(argv[5], argv[6], f_bound);

  const auto& [iProc,nProcs] = PITTS::internal::parallel::mpiProcInfo();

  // compress shape, s.t. first and last dimensions are bigger than max_r
  // first dimension is distributed over MPI processes
  std::vector<int> shape(d,n);
  const double nMin = std::max<double>(f_bound*max_r, 10.);
  while( shape.size() > 2 && shape.front() <= nMin*nProcs )
  {
    shape[1] *= shape[0];
    shape.erase(shape.begin());
  }
  while( shape.size() > 2 && shape.back() <= nMin )
  {
    n = shape.size();
    shape[n-2] *= shape[n-1];
    shape.pop_back();
  }

  // distribute first dimension
  {
    const auto& [first,last] = PITTS::internal::parallel::distribute(shape[0], {iProc,nProcs});
    shape[0] = last - first + 1;
  }

  std::size_t nTotal = 1;
  for(auto ni: shape)
    nTotal *= ni;


  PITTS::MultiVector<double> data(nTotal/shape.back(), shape.back());
  randomize(data);

  PITTS::MultiVector<double> X(nTotal/shape.back(), shape.back());
  PITTS::MultiVector<double> work;

  for(int iter = 0; iter < nIter; iter++)
  {
    copy(data, X);
    const auto TT = fromDense(X, work, shape, 1.e-8, max_r, true);
  }

  PITTS::finalize();

  return 0;
}

