// Copyright (c) 2023 German Aerospace Center (DLR), Institute for Software Technology, Germany
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>
#include "pitts_tensortrain_debug.hpp"
#include "pitts_tensortrain_to_dense.hpp"
#include "pitts_tensortrain_random.hpp"
#include "eigen_test_helper.hpp"

using vec = Eigen::VectorXd;
using namespace PITTS::debug;
using TensorTrain_double = PITTS::TensorTrain<double>;
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
}

TEST(PITTS_TensorTrain_debug, debug_example)
{
    TensorTrain_double ttX(3, 5), ttY(3, 5), ttZ(3, 5);
    ttX.setOnes();
    ttY.setOnes();
    ttZ.setOnes();

    // intended use case: check invariants with assert like this:
    ASSERT_NEAR(0., norm2(7.*ttX + 3.*ttY - 10.*ttZ), eps);
}

TEST(PITTS_TensorTrain_debug, scale_left)
{
    TensorTrain_double ttX(1, 50);
    randomize(ttX);

    TensorTrain_double ttY = 42.*ttX;

    ASSERT_NEAR(42*toVec(ttX), toVec(ttY), eps);
}


TEST(PITTS_TensorTrain_debug, scale_right)
{
    TensorTrain_double ttX(1, 50);
    randomize(ttX);

    TensorTrain_double ttY = ttX*42.;

    ASSERT_NEAR(42*toVec(ttX), toVec(ttY), eps);
}


TEST(PITTS_TensorTrain_debug, scale_right_inv)
{
    TensorTrain_double ttX(1, 50);
    randomize(ttX);

    TensorTrain_double ttY = ttX/42.;

    ASSERT_NEAR(toVec(ttX)/42, toVec(ttY), eps);
}


TEST(PITTS_TensorTrain_debug, add)
{
    TensorTrain_double ttX(1, 50), ttY(1, 50);
    randomize(ttX);
    randomize(ttY);

    TensorTrain_double ttZ = ttX + ttY;

    ASSERT_NEAR(toVec(ttX)+toVec(ttY), toVec(ttZ), eps);
}


TEST(PITTS_TensorTrain_debug, sub)
{
    TensorTrain_double ttX(1, 50), ttY(1, 50);
    randomize(ttX);
    randomize(ttY);

    TensorTrain_double ttZ = ttX - ttY;

    ASSERT_NEAR(toVec(ttX)-toVec(ttY), toVec(ttZ), eps);
}
