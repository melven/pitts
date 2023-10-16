// Copyright (c) 2020 German Aerospace Center (DLR), Institute for Software Technology, Germany
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>
#include "test_complex_helper.hpp"
#include "pitts_qubit_simulator.hpp"
#include "pitts_eigen.hpp"
#include <unsupported/Eigen/MatrixFunctions>

using namespace PITTS;

namespace
{
  constexpr auto eps = 1.e-10;

  using StateVec = FixedTensorTrain<std::complex<double>,2>;

  // helper function to convert a tensor train state vector to a plain dense vector
  Eigen::VectorXcd toDenseVector(const StateVec& TTv)
  {
    const auto nDims = TTv.nDims();
    const auto n = 1UL << nDims;
    Eigen::VectorXcd v(n);

    std::vector<int> idx_i(nDims);
    StateVec e_i(nDims);
    for(std::size_t i = 0; i < n; i++)
    {
      for(int j = 0; j < nDims; j++)
        idx_i[j] = (i & (1UL << j)) ? 1 : 0;
      e_i.setUnit(idx_i);

      v(i) = dot(e_i,TTv);
    }
    return v;
  }
}

TEST(PITTS_QubitSimulator, qubitIds_singleQubit)
{
  QubitSimulator qsim;

  // we don't have any qubits initially...
  ASSERT_THROW(qsim.isClassical(0), std::out_of_range);
  ASSERT_THROW(qsim.getClassicalValue(0), std::out_of_range);

  // allocate / deallocate
  qsim.allocateQubit(57);
  ASSERT_THROW(qsim.deallocateQubit(0), std::out_of_range);
  ASSERT_THROW(qsim.allocateQubit(57), std::invalid_argument);
  ASSERT_TRUE(qsim.isClassical(57));
  qsim.deallocateQubit(57);
  ASSERT_THROW(qsim.deallocateQubit(57), std::out_of_range);
}


TEST(PITTS_QubitSimulator, qubitIds_twoQubits)
{
  QubitSimulator qsim;

  // allocate / deallocate
  qsim.allocateQubit(57);
  qsim.allocateQubit(17);
  ASSERT_TRUE(qsim.isClassical(57));
  ASSERT_TRUE(qsim.isClassical(17));
  qsim.deallocateQubit(57);
  ASSERT_TRUE(qsim.isClassical(17));
  ASSERT_THROW(qsim.deallocateQubit(57), std::out_of_range);
  qsim.deallocateQubit(17);
}


TEST(PITTS_QubitSimulator, qubitIds_multipl)
{
  QubitSimulator qsim;

  // allocate / deallocate (with multiple swaps)
  qsim.allocateQubit(57);
  qsim.allocateQubit(17);
  qsim.allocateQubit(32);
  qsim.allocateQubit(5);
  qsim.deallocateQubit(17);
  qsim.deallocateQubit(57);
  ASSERT_THROW(qsim.allocateQubit(32), std::invalid_argument);
  ASSERT_THROW(qsim.deallocateQubit(57), std::out_of_range);
  qsim.allocateQubit(19);
  qsim.deallocateQubit(5);
  qsim.deallocateQubit(32);
  qsim.deallocateQubit(19);
}

TEST(PITTS_QubitSimulator, initializedToZero)
{
  QubitSimulator qsim;

  qsim.allocateQubit(57);
  qsim.allocateQubit(17);
  qsim.allocateQubit(32);
  qsim.allocateQubit(5);
  qsim.deallocateQubit(17);
  qsim.allocateQubit(0);

  EXPECT_EQ(false, qsim.getClassicalValue(57));
  EXPECT_EQ(false, qsim.getClassicalValue(32));
  ASSERT_THROW(qsim.getClassicalValue(17), std::out_of_range);
  EXPECT_EQ(false, qsim.getClassicalValue(5));
  EXPECT_EQ(false, qsim.getClassicalValue(0));
}

TEST(PITTS_QubitSimulator, collapse_alreadyClassicalZero)
{
  QubitSimulator qsim;

  qsim.allocateQubit(57);
  qsim.allocateQubit(17);
  qsim.allocateQubit(32);
  qsim.allocateQubit(5);

  // full collapse
  const std::vector<bool> result = {false,false,false,false};
  qsim.collapseWavefunction({5,32,17,57}, result);

  // partial collapse
  const std::vector<bool> result2 = {false,false};
  qsim.collapseWavefunction({32,17}, result2);

  // non-existing ids
  const std::vector<bool> tmp(3);
  ASSERT_THROW(qsim.collapseWavefunction({1,2,3}, tmp), std::out_of_range);

  // incorrect dimensions
  ASSERT_THROW(qsim.collapseWavefunction({32,17}, tmp), std::invalid_argument);

  // invalid collapse (impossible values)
  const std::vector<bool> result3 = {true,false};
  ASSERT_THROW(qsim.collapseWavefunction({32,5}, result3), std::invalid_argument);
}

TEST(PITTS_QubitSimulator, getProbability_alreadyClassicalZero)
{
  QubitSimulator qsim;

  qsim.allocateQubit(57);
  qsim.allocateQubit(17);
  qsim.allocateQubit(32);
  qsim.allocateQubit(5);

  // full probability
  const std::vector<bool> result_ref = {false,false,false,false};
  EXPECT_NEAR(1., qsim.getProbability({5,32,17,57}, result_ref), eps);

  const std::vector<bool> other_result = {true,false,true,false};
  EXPECT_NEAR(0., qsim.getProbability({5,32,17,57}, other_result), eps);

  // partial probability
  const std::vector<bool> result2_ref = {false,false};
  EXPECT_NEAR(1., qsim.getProbability({32,17}, result2_ref), eps);

  const std::vector<bool> other_result2 = {true,false};
  EXPECT_NEAR(0., qsim.getProbability({32,17}, other_result2), eps);

  // non-existing ids
  ASSERT_THROW(qsim.getProbability({32,2}, {true,false}), std::out_of_range);

  // inconsistent input dimensions
  ASSERT_THROW(qsim.getProbability({32,2}, {true,false,true}), std::invalid_argument);
}

TEST(PITTS_QubitSimulator, measureQubits_alreadyClassicalZero)
{
  QubitSimulator qsim;

  qsim.allocateQubit(57);
  qsim.allocateQubit(17);
  qsim.allocateQubit(32);
  qsim.allocateQubit(5);

  // full measurement
  const auto result = qsim.measureQubits({5,32,17,57});
  const std::vector<bool> result_ref = {false,false,false,false};
  EXPECT_EQ(result_ref, result);

  // partial measurement
  const auto result2 = qsim.measureQubits({32,17});
  const std::vector<bool> result2_ref = {false,false};
  EXPECT_EQ(result2_ref, result2);

  // non-existing ids
  ASSERT_THROW(qsim.measureQubits({32,2}), std::out_of_range);
}

TEST(PITTS_QubitSimulator, applySingleQubitGate_classicalBitFlip)
{
  QubitSimulator::Matrix2 flipGate;
  flipGate[0][0] = 0;
  flipGate[0][1] = 1;
  flipGate[1][0] = 1;
  flipGate[1][1] = 0;

  QubitSimulator qsim;

  // allocate some qubits
  qsim.allocateQubit(1);
  qsim.allocateQubit(3);
  qsim.allocateQubit(5);
  qsim.allocateQubit(7);

  EXPECT_EQ(false, qsim.getClassicalValue(1));
  EXPECT_EQ(false, qsim.getClassicalValue(3));
  EXPECT_EQ(false, qsim.getClassicalValue(5));
  EXPECT_EQ(false, qsim.getClassicalValue(7));

  qsim.applySingleQubitGate(3, flipGate);

  EXPECT_EQ(false, qsim.getClassicalValue(1));
  EXPECT_EQ(true, qsim.getClassicalValue(3));
  EXPECT_EQ(false, qsim.getClassicalValue(5));
  EXPECT_EQ(false, qsim.getClassicalValue(7));

  // requires reordering of the internal state vector!
  qsim.allocateQubit(9);
  qsim.deallocateQubit(1);

  EXPECT_EQ(false, qsim.getClassicalValue(9));
  EXPECT_EQ(true, qsim.getClassicalValue(3));
  EXPECT_EQ(false, qsim.getClassicalValue(5));
  EXPECT_EQ(false, qsim.getClassicalValue(7));

  const auto result = qsim.measureQubits({9,5,7,3});
  const std::vector<bool> result_ref = {false,false,false,true};
  EXPECT_EQ(result_ref, result);

  qsim.applySingleQubitGate(7, flipGate);
  qsim.applySingleQubitGate(5, flipGate);
  qsim.applySingleQubitGate(3, flipGate);
  qsim.applySingleQubitGate(9, flipGate);

  const auto result2 = qsim.measureQubits({9,5,7,3});
  const std::vector<bool> result2_ref = {true,true,true,false};
  EXPECT_EQ(result2_ref, result2);
}

TEST(PITTS_QubitSimulator, applyTwoQubitGate_classicalBitFlip)
{
  QubitSimulator::Matrix4 flipBothGate = {};
  flipBothGate[0][3] = 1;
  flipBothGate[1][2] = 1;
  flipBothGate[2][1] = 1;
  flipBothGate[3][0] = 1;

  QubitSimulator qsim;

  // allocate some qubits
  qsim.allocateQubit(1);
  qsim.allocateQubit(3);
  qsim.allocateQubit(5);
  qsim.allocateQubit(7);
  qsim.allocateQubit(9);

  EXPECT_EQ(false, qsim.getClassicalValue(1));
  EXPECT_EQ(false, qsim.getClassicalValue(3));
  EXPECT_EQ(false, qsim.getClassicalValue(5));
  EXPECT_EQ(false, qsim.getClassicalValue(7));
  EXPECT_EQ(false, qsim.getClassicalValue(9));

  qsim.applyTwoQubitGate(3, 5, flipBothGate);

  EXPECT_EQ(false, qsim.getClassicalValue(1));
  EXPECT_EQ(true, qsim.getClassicalValue(3));
  EXPECT_EQ(true, qsim.getClassicalValue(5));
  EXPECT_EQ(false, qsim.getClassicalValue(7));
  EXPECT_EQ(false, qsim.getClassicalValue(9));

  qsim.applyTwoQubitGate(5, 7, flipBothGate);

  EXPECT_EQ(false, qsim.getClassicalValue(1));
  EXPECT_EQ(true, qsim.getClassicalValue(3));
  EXPECT_EQ(false, qsim.getClassicalValue(5));
  EXPECT_EQ(true, qsim.getClassicalValue(7));
  EXPECT_EQ(false, qsim.getClassicalValue(9));

  qsim.applyTwoQubitGate(3, 7, flipBothGate);

  EXPECT_EQ(false, qsim.getClassicalValue(1));
  EXPECT_EQ(false, qsim.getClassicalValue(3));
  EXPECT_EQ(false, qsim.getClassicalValue(5));
  EXPECT_EQ(false, qsim.getClassicalValue(7));
  EXPECT_EQ(false, qsim.getClassicalValue(9));

  qsim.applyTwoQubitGate(9, 1, flipBothGate);

  EXPECT_EQ(true, qsim.getClassicalValue(1));
  EXPECT_EQ(false, qsim.getClassicalValue(3));
  EXPECT_EQ(false, qsim.getClassicalValue(5));
  EXPECT_EQ(false, qsim.getClassicalValue(7));
  EXPECT_EQ(true, qsim.getClassicalValue(9));
}

TEST(PITTS_QubitSimulator, applySingleAndTwoQubitGates_simpleBellState)
{
  QubitSimulator::Matrix2 hadamardGate;
  hadamardGate[0][0] = 1./std::sqrt(2.);
  hadamardGate[0][1] = 1./std::sqrt(2.);
  hadamardGate[1][0] = 1./std::sqrt(2.);
  hadamardGate[1][1] = -1./std::sqrt(2.);

  QubitSimulator::Matrix4 cnotGate = {};
  cnotGate[0][0] = 1;
  cnotGate[1][1] = 1;
  cnotGate[2][3] = 1;
  cnotGate[3][2] = 1;

  QubitSimulator qsim;

  // allocate some qubits
  qsim.allocateQubit(1);
  qsim.allocateQubit(3);
  qsim.allocateQubit(5);
  qsim.allocateQubit(7);
  qsim.allocateQubit(9);

  EXPECT_EQ(false, qsim.getClassicalValue(1));
  EXPECT_EQ(false, qsim.getClassicalValue(3));
  EXPECT_EQ(false, qsim.getClassicalValue(5));
  EXPECT_EQ(false, qsim.getClassicalValue(7));
  EXPECT_EQ(false, qsim.getClassicalValue(9));

  qsim.applySingleQubitGate(5, hadamardGate);

  EXPECT_EQ(true, qsim.isClassical(1));
  EXPECT_EQ(true, qsim.isClassical(3));
  EXPECT_EQ(false, qsim.isClassical(5));
  EXPECT_EQ(true, qsim.isClassical(7));
  EXPECT_EQ(true, qsim.isClassical(9));

  // create entangled state between 3 and 5
  qsim.applyTwoQubitGate(3, 5, cnotGate);

  EXPECT_EQ(true, qsim.isClassical(1));
  EXPECT_EQ(false, qsim.isClassical(3));
  EXPECT_EQ(false, qsim.isClassical(5));
  EXPECT_EQ(true, qsim.isClassical(7));
  EXPECT_EQ(true, qsim.isClassical(9));

  // check probabilities
  ASSERT_NEAR(0.5, qsim.getProbability({3,5}, {false,false}), eps);
  ASSERT_NEAR(0.5, qsim.getProbability({3,5}, {true,true}), eps);

  // cannot deallocate entangled state...
  ASSERT_THROW(qsim.deallocateQubit(3), std::invalid_argument);

  // collapse to classical state
  qsim.collapseWavefunction({3,5}, {false,false});

  EXPECT_EQ(false, qsim.getClassicalValue(1));
  EXPECT_EQ(false, qsim.getClassicalValue(3));
  EXPECT_EQ(false, qsim.getClassicalValue(5));
  EXPECT_EQ(false, qsim.getClassicalValue(7));
  EXPECT_EQ(false, qsim.getClassicalValue(9));

  // check that the scaling is still correct after the collapse
  ASSERT_NEAR(1., qsim.getProbability({1,3,5,7,9}, {false,false,false,false,false}), eps);

  // same as before but try other collapse
  qsim.applySingleQubitGate(5, hadamardGate);
  qsim.applyTwoQubitGate(3, 5, cnotGate);
  ASSERT_THROW(qsim.collapseWavefunction({3,5}, {true,false}), std::invalid_argument);
  ASSERT_THROW(qsim.collapseWavefunction({3,5}, {false,true}), std::invalid_argument);
  qsim.collapseWavefunction({3,5}, {true,true});

  EXPECT_EQ(false, qsim.getClassicalValue(1));
  EXPECT_EQ(true, qsim.getClassicalValue(3));
  EXPECT_EQ(true, qsim.getClassicalValue(5));
  EXPECT_EQ(false, qsim.getClassicalValue(7));
  EXPECT_EQ(false, qsim.getClassicalValue(9));

  ASSERT_NEAR(1., qsim.getProbability({1,3,5,7,9}, {false,true,true,false,false}), eps);
}

TEST(PITTS_QubitSimulator, emulateTimeEvolution)
{
  QubitSimulator qsim;

  // allocate some qubits
  qsim.allocateQubit(1);
  qsim.allocateQubit(3);
  qsim.allocateQubit(5);
  qsim.allocateQubit(7);
  qsim.allocateQubit(9);

  // construct some arbitrary Hamiltonian
  const std::vector<QubitSimulator::QubitId> ids = {3,5,7};
  QubitSimulator::Matrix2 h3;
  h3[0][0] = 0.;
  h3[0][1] = 0.77;
  h3[1][0] = 0.77;
  h3[1][1] = 0.;

  QubitSimulator::Matrix2 h5;
  h5[0][0] = 1.55;
  h5[0][1] = 0.;
  h5[1][0] = 0.;
  h5[1][1] = -1.55;

  QubitSimulator::Matrix2 h7;
  h7[0][0] = 0.;
  h7[0][1] = std::complex<double>(0., -1.33);
  h7[1][0] = std::complex<double>(0., 1.33);
  h7[1][1] = 0.;
  const std::vector<QubitSimulator::Matrix2> terms = {h3,h5,h7};

  // compose complete Hamiltonian matrix
  using Matrix32 = Eigen::Matrix<std::complex<double>, 32, 32>;
  Matrix32 H;
  {
    StateVec col_i(5);
    std::vector<int> idx_i(5);
    for(std::size_t i = 0; i < 32; i++)
    {
      for(int j = 0; j < 5; j++)
        idx_i[j] = (i & (1UL << j)) ? 1 : 0;

      col_i.setUnit(idx_i);

      PITTS::apply(col_i.editableSubTensors()[1], h3);
      PITTS::apply(col_i.editableSubTensors()[2], h5);
      PITTS::apply(col_i.editableSubTensors()[3], h7);
      H.col(i) = toDenseVector(col_i);
    }
  }

  // H should be self-adjoint
  ASSERT_NEAR(0., (H-H.adjoint()).norm(), eps);

  // helper variable for complex number i
  static constexpr auto i = std::complex<double>(0, 1);

  // get the initial state
  auto x0 = toDenseVector(qsim.getWaveFunction());

  // no change
  qsim.emulateTimeEvolution(0., ids, terms);
  auto x = toDenseVector(qsim.getWaveFunction());
  auto x_ref = x0;
  EXPECT_NEAR(0., (x-x_ref).norm(), eps);

  // forward
  qsim.emulateTimeEvolution(0.3, ids, terms);
  x = toDenseVector(qsim.getWaveFunction());
  x_ref = (i*0.3*H).exp() * x0;
  EXPECT_NEAR(0., (x-x_ref).norm(), eps);

  // backward
  qsim.emulateTimeEvolution(-0.17, ids, terms);
  x = toDenseVector(qsim.getWaveFunction());
  x_ref = (i*0.13*H).exp() * x0;
  EXPECT_NEAR(0., (x-x_ref).norm(), eps);


  // use a non-classical state to make this more interesting!
  {
    QubitSimulator::Matrix2 phaseGate;
    phaseGate[0][0] = 1.;
    phaseGate[0][1] = 0.;
    phaseGate[1][0] = 0.;
    phaseGate[1][1] = i;

    QubitSimulator::Matrix2 hadamardGate;
    hadamardGate[0][0] = 1./std::sqrt(2.);
    hadamardGate[0][1] = 1./std::sqrt(2.);
    hadamardGate[1][0] = 1./std::sqrt(2.);
    hadamardGate[1][1] = -1./std::sqrt(2.);

    QubitSimulator::Matrix4 cnotGate = {};
    cnotGate[0][0] = 1;
    cnotGate[1][1] = 1;
    cnotGate[2][3] = 1;
    cnotGate[3][2] = 1;

    qsim.applySingleQubitGate(1, phaseGate);
    qsim.applySingleQubitGate(5, hadamardGate);
    qsim.applySingleQubitGate(9, hadamardGate);
    qsim.applySingleQubitGate(9, phaseGate);
    qsim.applyTwoQubitGate(7, 9, cnotGate);
    qsim.applyTwoQubitGate(3, 5, cnotGate);
    qsim.applyTwoQubitGate(1, 3, cnotGate);
    qsim.applySingleQubitGate(5, phaseGate);
    qsim.applyTwoQubitGate(5, 7, cnotGate);
  }

  // get the initial state
  x0 = toDenseVector(qsim.getWaveFunction());

  // no change
  qsim.emulateTimeEvolution(0., ids, terms);
  x = toDenseVector(qsim.getWaveFunction());
  x_ref = x0;
  EXPECT_NEAR(0., (x-x_ref).norm(), eps);

  // forward
  qsim.emulateTimeEvolution(0.3, ids, terms);
  x = toDenseVector(qsim.getWaveFunction());
  x_ref = (i*0.3*H).exp() * x0;
  EXPECT_NEAR(0., (x-x_ref).norm(), eps);

  // backward
  qsim.emulateTimeEvolution(-0.17, ids, terms);
  x = toDenseVector(qsim.getWaveFunction());
  x_ref = (i*0.13*H).exp() * x0;
  EXPECT_NEAR(0., (x-x_ref).norm(), eps);
}

TEST(PITTS_QubitSimulator, getExpectationValue)
{
  QubitSimulator qsim;

  // allocate some qubits
  qsim.allocateQubit(1);
  qsim.allocateQubit(3);
  qsim.allocateQubit(5);
  qsim.allocateQubit(7);
  qsim.allocateQubit(9);

  // construct some arbitrary Hamiltonian
  QubitSimulator::Matrix2 up = {};
  up[0][0] = 1.;

  QubitSimulator::Matrix2 down = {};
  down[1][1] = 1.;

  QubitSimulator::Matrix2 upDoubled = {};
  upDoubled[0][0] = 2.;

  // initially everything points up
  EXPECT_NEAR(1., qsim.getExpectationValue({1}, {up}), eps);
  EXPECT_NEAR(1., qsim.getExpectationValue({3}, {up}), eps);
  EXPECT_NEAR(1., qsim.getExpectationValue({5}, {up}), eps);
  EXPECT_NEAR(1., qsim.getExpectationValue({1,9}, {up,down}), eps);
  EXPECT_NEAR(3., qsim.getExpectationValue({3,5,7}, {up,up,up}), eps);


  // use a non-classical state to make this more interesting!
  {
    // helper variable for complex number i
    static constexpr auto i = std::complex<double>(0, 1);

    QubitSimulator::Matrix2 phaseGate;
    phaseGate[0][0] = 1.;
    phaseGate[0][1] = 0.;
    phaseGate[1][0] = 0.;
    phaseGate[1][1] = i;

    QubitSimulator::Matrix2 hadamardGate;
    hadamardGate[0][0] = 1./std::sqrt(2.);
    hadamardGate[0][1] = 1./std::sqrt(2.);
    hadamardGate[1][0] = 1./std::sqrt(2.);
    hadamardGate[1][1] = -1./std::sqrt(2.);

    QubitSimulator::Matrix4 cnotGate = {};
    cnotGate[0][0] = 1;
    cnotGate[1][1] = 1;
    cnotGate[2][3] = 1;
    cnotGate[3][2] = 1;

    qsim.applySingleQubitGate(1, phaseGate);
    qsim.applySingleQubitGate(5, hadamardGate);
    qsim.applySingleQubitGate(9, hadamardGate);
    qsim.applySingleQubitGate(9, phaseGate);
    qsim.applyTwoQubitGate(7, 9, cnotGate);
    qsim.applyTwoQubitGate(3, 5, cnotGate);
    qsim.applyTwoQubitGate(1, 3, cnotGate);
    qsim.applySingleQubitGate(5, phaseGate);
    qsim.applyTwoQubitGate(5, 7, cnotGate);
  }

  EXPECT_NEAR(qsim.getProbability({1}, {false}), qsim.getExpectationValue({1}, {up}), eps);
  EXPECT_NEAR(qsim.getProbability({3}, {false}), qsim.getExpectationValue({3}, {up}), eps);
  EXPECT_NEAR(qsim.getProbability({5}, {false}), qsim.getExpectationValue({5}, {up}), eps);
  EXPECT_NEAR(qsim.getProbability({7}, {false}), qsim.getExpectationValue({7}, {up}), eps);
  EXPECT_NEAR(qsim.getProbability({9}, {false}), qsim.getExpectationValue({9}, {up}), eps);

  EXPECT_NEAR(qsim.getProbability({1}, {true}), qsim.getExpectationValue({1}, {down}), eps);
  EXPECT_NEAR(qsim.getProbability({3}, {true}), qsim.getExpectationValue({3}, {down}), eps);
  EXPECT_NEAR(qsim.getProbability({5}, {true}), qsim.getExpectationValue({5}, {down}), eps);
  EXPECT_NEAR(qsim.getProbability({7}, {true}), qsim.getExpectationValue({7}, {down}), eps);
  EXPECT_NEAR(qsim.getProbability({9}, {true}), qsim.getExpectationValue({9}, {down}), eps);

  EXPECT_NEAR(2*qsim.getProbability({1}, {false}), qsim.getExpectationValue({1}, {upDoubled}), eps);
}
