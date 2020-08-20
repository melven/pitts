#include "pitts_parallel.hpp"
#include "pitts_common.hpp"
#include "pitts_qubit_simulator.hpp"
#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

namespace
{
  std::ostream& operator<<(std::ostream& s, const std::vector<int>& v)
  {
    for(auto i: v)
      s << " " << i;
    return s;
  }
}


int main(int argc, char* argv[])
{
  PITTS::initialize(&argc, &argv);

  PITTS::QubitSimulator::Matrix2 phaseGate;
  phaseGate[0][0] = 1.;
  phaseGate[0][1] = 0.;
  phaseGate[1][0] = 0.;
  phaseGate[1][1] = std::complex<double>(std::sin(0.11), std::cos(0.11));

  PITTS::QubitSimulator::Matrix2 hadamardGate;
  hadamardGate[0][0] = 1./std::sqrt(2.);
  hadamardGate[0][1] = 1./std::sqrt(2.);
  hadamardGate[1][0] = 1./std::sqrt(2.);
  hadamardGate[1][1] = -1./std::sqrt(2.);

  PITTS::QubitSimulator::Matrix4 cnotGate = {};
  cnotGate[0][0] = 1;
  cnotGate[1][1] = 1;
  cnotGate[2][3] = 1;
  cnotGate[3][2] = 1;

  PITTS::QubitSimulator qsim;
  using QubitId = PITTS::QubitSimulator::QubitId;
  const QubitId nQ = 100;
  for(QubitId iQ = 0; iQ < nQ; iQ++)
    qsim.allocateQubit(iQ);

  double wtime = omp_get_wtime();

  // just single qubit gates
  const int nIter = 15;
  for(int iter = 0; iter < nIter; iter++)
  {
    if( iter % 2 == 0 )
      for(QubitId iQ = 0; iQ < nQ; iQ+=2)
        qsim.applySingleQubitGate(iQ, hadamardGate);
    else
      for(QubitId iQ = 0; iQ < nQ; iQ+=3)
        qsim.applySingleQubitGate(iQ, phaseGate);

    for(QubitId iQ = 1; iQ+1 < nQ; iQ+=2)
      qsim.applyTwoQubitGate(iQ, 0, cnotGate);

    for(QubitId iQ = 0; iQ+1 < nQ; iQ+=5)
      qsim.applyTwoQubitGate(iQ+1, 7, cnotGate);

    std::cout << "TT ranks: " << qsim.getWaveFunction().getTTranks() << "\n";
  }

  wtime = omp_get_wtime() - wtime;
  std::cout << "Run time: " << wtime << "\n";

  PITTS::finalize();

  return 0;
}
