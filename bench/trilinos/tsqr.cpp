#include <Teuchos_RCP.hpp>
#include <Tpetra_Core.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Tpetra_TsqrAdaptor.hpp>
#include <charconv>
#include <iostream>
#include <exception>
#include <omp.h>

int main(int argc, char*argv[])
{
  using MultiVector = Tpetra::MultiVector<double>;
  using ConstMap = const MultiVector::map_type;
  using TSQR = Tpetra::TsqrAdaptor<MultiVector>;
  using DenseMat = TSQR::dense_matrix_type;

  if( argc != 4 )
    throw std::invalid_argument("Requires 3 arguments (n m nIter)!");

  long long n = 0, m = 0;
  int nIter = 0;
  std::from_chars(argv[1], argv[2], n);
  std::from_chars(argv[2], argv[3], m);
  std::from_chars(argv[3], argv[4], nIter);


  Teuchos::RCP<ConstMap> map(new ConstMap(n, 0, Tpetra::getDefaultComm()));


  double wtime = omp_get_wtime();
  MultiVector X_in(map, m), X(map, m), Q(map, m);
  wtime = omp_get_wtime() - wtime;
  std::cout << "wtime alloc: " << wtime << "\n";

  wtime = omp_get_wtime();
  X_in.randomize();
  wtime = omp_get_wtime() - wtime;
  std::cout << "wtime randomize: " << wtime << "\n";

  TSQR tsqr;

  //std::cout << "TSQR parameters:\n";
  //Teuchos::RCP<Teuchos::ParameterList> params(new Teuchos::ParameterList);
  //const auto params = tsqr.getValidParameters();
  //params->set("Cache Size Hint", 160);
  //params->print(std::cout);
  //tsqr.setParameterList(params);

  DenseMat R(m, m);

  double wtime_copy = 0, wtime_tsqr = 0;
  for(int i = 0; i < nIter; i++)
  {
    wtime = omp_get_wtime();
    X.assign(X_in);
    wtime_copy += omp_get_wtime() - wtime;

    wtime = omp_get_wtime();
    tsqr.factorExplicit(X, Q, R);
    wtime_tsqr += omp_get_wtime() - wtime;
  }

  std::cout << "wtime copy: " << wtime_copy << "\n";
  std::cout << "wtime tsqr: " << wtime_tsqr << "\n";

  return 0;
}
