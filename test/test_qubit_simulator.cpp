#include <gtest/gtest.h>
#include "test_complex_helper.hpp"
#include "pitts_qubit_simulator.hpp"

using namespace PITTS;

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

TEST(PITTS_QubitSimulator, measure_alreadyClassicalZero)
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
  constexpr auto eps = 1.e-10;

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
