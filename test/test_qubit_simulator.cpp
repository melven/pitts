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
