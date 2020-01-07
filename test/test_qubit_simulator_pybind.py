import pitts_py
from math import sqrt

qsim = pitts_py.QubitSimulator()

hadamardGate = [[1/sqrt(2.),1/sqrt(2.)],[1/sqrt(2.),-1/sqrt(2.)]]
cnotGate = [[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]

for i in range(10):
  qsim.allocate_qubit(7)
  qsim.allocate_qubit(9)

  print(qsim.get_classical_value(7))

  qsim.apply_single_qubit_gate(9, hadamardGate)
  qsim.apply_two_qubit_gate(7, 9, cnotGate)

  print(qsim.measure_qubits((7,9)))

  qsim.deallocate_qubit(9)
  qsim.deallocate_qubit(7)
