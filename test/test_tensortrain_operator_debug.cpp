// Copyright (c) 2023 German Aerospace Center (DLR), Institute for Software Technology, Germany
//
// SPDX-License-Identifier: BSD-3-Clause

#include <vector>
#include <gtest/gtest.h>
#include "pitts_tensortrain_operator_debug.hpp"
#include "pitts_tensortrain_debug.hpp"
#include "pitts_tensortrain_to_dense.hpp"
#include "pitts_tensortrain_random.hpp"
#include "eigen_test_helper.hpp"

using vec = Eigen::VectorXd;
using mat = Eigen::MatrixXd;
using namespace PITTS::debug;
using TensorTrain_double = PITTS::TensorTrain<double>;
using TensorTrainOperator_double = PITTS::TensorTrainOperator<double>;
constexpr double eps = 1.e-10;

// anonymous namespace with helper functions
namespace
{
    vec toVec(const TensorTrain_double& ttX)
    {
        if( ttX.dimensions().size() != 1 )
            throw std::invalid_argument("Number of dimensions must be 1!");
        
        vec vX(ttX.dimensions()[0]);
        toDense(ttX, vX.data(), vX.data()+vX.size());
        return vX;
    }

    mat toMat(const TensorTrainOperator_double& ttOpA)
    {
        if( ttOpA.row_dimensions().size() != 1 )
            throw std::invalid_argument("Number of dimensions must be 1!");
        
        const int n = ttOpA.row_dimensions()[0];
        const int m = ttOpA.column_dimensions()[0];
        std::vector<double> rawA(n*m);
        toDense(ttOpA.tensorTrain(), rawA.begin(), rawA.end());
        return Eigen::Map<mat, Eigen::ColMajor>(rawA.data(), n, m);
    }
}

TEST(PITTS_TensorTrainOperator_debug, debug_example)
{
    TensorTrain_double ttX(3, 5), ttY(3, 5), ttZ(3, 5);
    ttX.setOnes();
    ttY.setOnes();
    ttZ.setOnes();
    TensorTrainOperator_double ttOpA(3, 5, 5);
    ttOpA.setEye();

    // intended use case: check invariants with assert like this:
    ASSERT_NEAR(0., norm2(7.*ttOpA*ttX + 3.*transpose(ttOpA)*ttY - 10.*ttZ), eps);
}

TEST(PITTS_TensorTrainOperator_debug, scale_left)
{
    TensorTrainOperator_double ttOpA(1, 10, 5);
    randomize(ttOpA);

    TensorTrainOperator_double ttOpB = 42.*ttOpA;

    ASSERT_NEAR(42*toMat(ttOpA), toMat(ttOpB), eps);
}


TEST(PITTS_TensorTrainOperator_debug, scale_right)
{
    TensorTrainOperator_double ttOpA(1, 10, 5);
    randomize(ttOpA);

    TensorTrainOperator_double ttOpB = ttOpA*42.;

    ASSERT_NEAR(42*toMat(ttOpA), toMat(ttOpB), eps);
}


TEST(PITTS_TensorTrainOperator_debug, scale_right_inv)
{
    TensorTrainOperator_double ttOpA(1, 10, 5);
    randomize(ttOpA);

    TensorTrainOperator_double ttOpB = ttOpA/42.;

    ASSERT_NEAR(toMat(ttOpA)/42, toMat(ttOpB), eps);
}


TEST(PITTS_TensorTrainOperator_debug, add)
{
    TensorTrainOperator_double ttOpA(1, 10, 5), ttOpB(1, 10, 5);
    randomize(ttOpA);
    randomize(ttOpB);

    TensorTrainOperator_double ttOpC = ttOpA + ttOpB;

    ASSERT_NEAR(toMat(ttOpA)+toMat(ttOpB), toMat(ttOpC), eps);
}


TEST(PITTS_TensorTrainOperator_debug, sub)
{
    TensorTrainOperator_double ttOpA(1, 10, 5), ttOpB(1, 10, 5);
    randomize(ttOpA);
    randomize(ttOpB);

    TensorTrainOperator_double ttOpC = ttOpA - ttOpB;

    ASSERT_NEAR(toMat(ttOpA)-toMat(ttOpB), toMat(ttOpC), eps);
}


TEST(PITTS_TensorTrainOperator_debug, transpose)
{
    TensorTrainOperator_double ttOpA(1, 10, 5);
    randomize(ttOpA);

    TensorTrainOperator_double ttOpB = transpose(ttOpA);

    ASSERT_NEAR(toMat(ttOpA).transpose(), toMat(ttOpB), eps);
}


TEST(PITTS_TensorTrainOperator_debug, apply)
{
    TensorTrainOperator_double ttOpA(1, 10, 5);
    randomize(ttOpA);
    TensorTrain_double ttX(1, 5);
    randomize(ttX);

    TensorTrain_double ttY = ttOpA * ttX;

    ASSERT_NEAR(toMat(ttOpA) * toVec(ttX), toVec(ttY), eps);
}


TEST(PITTS_TensorTrainOperator_debug, apply_op)
{
    TensorTrainOperator_double ttOpA(1, 10, 5), ttOpB(1, 5, 7);
    randomize(ttOpA);
    randomize(ttOpB);

    TensorTrainOperator_double ttOpC = ttOpA * ttOpB;

    ASSERT_NEAR(toMat(ttOpA) * toMat(ttOpB), toMat(ttOpC), eps);
}
