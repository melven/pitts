// Copyright (c) 2022 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Manuel Joey Becklas
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>
#include "pitts_tensor3.hpp"
#include "pitts_tensortrain.hpp"
#include "pitts_tensortrain_random.hpp"
#include "pitts_tensortrain_axpby_plain.hpp"
#include "pitts_tensortrain_axpby_normalized.hpp"
#include "pitts_tensortrain_normalize.hpp"
#include "pitts_tensortrain_norm.hpp"
#include "pitts_tensortrain_to_dense.hpp"
#include "eigen_test_helper.hpp"
#include "pitts_multivector_eigen_adaptor.hpp"

using namespace PITTS;


/**
 * @brief Helper function for testing axpby_normalized implementation.
 * 
 * @param alpha 
 * @param TTx 
 * @param beta 
 * @param TTy 
 * @param accuracy 
 */
static auto check_axpby(double alpha, const TensorTrain<double>& TTx, double beta, TensorTrain<double>& TTy, double accuracy = 10*std::sqrt(std::numeric_limits<double>::epsilon()))
{
    TensorTrain<double> _TTy_(TTy);

    // if x is not orthogonal, function was called with wrong parameters
    EXPECT_NE(TTx.isOrthogonal(), TT_Orthogonality::none);

    const double _gamma_ = internal::axpby_normalized(alpha, TTx, beta, _TTy_, accuracy, 0x7fffffff);
    const double gamma = internal::axpby_plain(alpha, TTx, beta, TTy, accuracy);
    const double _norm_ = norm2(_TTy_);

    const double absolute_accuracy = std::max(1.e-60, accuracy * norm2(TTy)); // scale by norm of "correct" result
    
    MultiVector<double> _y_, y;
    toDense(TTy, y);
    toDense(_TTy_, _y_);

    // checks
    EXPECT_NEAR(_gamma_, gamma, absolute_accuracy);
    if (alpha*beta != 0.) {EXPECT_NEAR(_norm_, 1.0, absolute_accuracy);}
    EXPECT_NEAR(ConstEigenMap(_y_), ConstEigenMap(y), absolute_accuracy);
    EXPECT_TRUE(internal::is_normalized(_TTy_, _TTy_.isOrthogonal(), accuracy));
    //EXPECT_NE(_TTy_.isOrthogonal(), TTx.isOrthogonal()); not required (even tho it is the case)

    return std::make_pair(gamma, _TTy_);
}


/** 
 * @brief Helper function: Initializes cores in [first_idx, last_idx) of tensor train randomly.
 * 
 * @param tt            tensor train object
 * @param max           maximal absolute value to initialize with
 * @param first_idx     first index
 * @param last_idx      last index: -1 <==> last
 */
static void rand_init(TensorTrain<double>& tt, double max = 1.0, int first_idx = 0, int last_idx = -1)
{
    if (last_idx == -1) last_idx = tt.dimensions().size();
   // initialize TT with (pseudo-)random numbers in range [0, max]
    for (int k = first_idx; k < last_idx; k++)
    {
        Tensor3<double> core(tt.subTensor(k).r1(), tt.subTensor(k).n(), tt.subTensor(k).r2());
        for (int i2 = 0; i2 < core.r2(); i2++)
            for (int j = 0; j < core.n(); j++)
                for (int i1  = 0; i1 < core.r1(); i1++)
                    core(i1, j, i2) = (double)std::rand() / RAND_MAX * max;
        core = tt.setSubTensor(k, std::move(core));
    }
    
}

/**
 * @brief Helper Function: Left-orthogonalizes cores in [first_idx, last_idx - 1) of tensor train.
 * 
 * @param tt            tensor train object
 * @param first_idx     first index
 * @param last_idx      last index: -1 <==> last
 */
static void left_ortho(TensorTrain<double>& tt, int first_idx = 0, int last_idx = -1)
{
    if (last_idx == -1) last_idx = tt.dimensions().size();
    // left-orthogonalize TT
    internal::leftNormalize_range<double>(tt, first_idx, last_idx - 1, 0);
}

/**
 * @brief Helper Function: Right-orthogonalizes cores in [first_idx + 1, last_idx) of tensor train.
 * 
 * @param tt            tensor train object
 * @param first_idx     first index
 * @param last_idx      last index: -1 <==> last
 */
static void right_ortho(TensorTrain<double>& tt, int first_idx = 0, int last_idx = -1)
{
    if (last_idx == -1) last_idx = tt.dimensions().size();
    // right-orthogonalize TT
    internal::rightNormalize_range<double>(tt, first_idx, last_idx - 1, 0);
}


static void printTTranks(const TensorTrain<double>& tt, const char* name)
{
    std::vector<int> ranks = tt.getTTranks();
    std::cout << "tensor train ranks of " << name << ": ";
    for (int rank : ranks) 
        std::cout << rank << "\t";
    std::cout << std::endl;
}


/****************************************************************************************************
 * LEFT
 ***************************************************************************************************/

TEST(PITTS_TensorTrain_axpby_normalized, left_order1)
{
    // TTx, TTy:
    //  o
    //  |
    //  11

    TensorTrain<double> TTx(1, 11), TTy(1, 11);

    randomize(TTx);
    randomize(TTy);
    left_ortho(TTx);

    check_axpby(1.0, TTx, 1.0, TTy);
    check_axpby(1.0, TTx, -2.0, TTy);
    check_axpby(3.0, TTx, 1.0, TTy);
    check_axpby(-4.0, TTx, -5.0, TTy);
}

TEST(PITTS_TensorTrain_axpby_normalized, left_order2_0)
{
    // TTx, TTy:
    //    3
    //  o -- o
    //  |    |
    //  4    4

    TensorTrain<double> TTx(2, 4, 3), TTy(2, 4, 3);

    randomize(TTx);
    randomize(TTy);
    left_ortho(TTx);

    check_axpby(1.0, TTx, 1.0, TTy);
    check_axpby(1.0, TTx, -2.0, TTy);
    check_axpby(3.0, TTx, 1.0, TTy);
    check_axpby(-4.0, TTx, -5.0, TTy);
}

TEST(PITTS_TensorTrain_axpby_normalized, left_order2_1)
{
    // TTx:         TTy:
    //    4            2
    //  o -- o       o -- o
    //  |    |       |    |
    //  4    4       4    4

    TensorTrain<double> TTx(2, 4, 4), TTy(2, 4, 2);

    randomize(TTx);
    randomize(TTy);
    left_ortho(TTx);

    check_axpby(1.0, TTx, 1.0, TTy);
    check_axpby(1.0, TTx, -2.0, TTy);
    check_axpby(3.0, TTx, 1.0, TTy);
    check_axpby(-4.0, TTx, -5.0, TTy);
}

TEST(PITTS_TensorTrain_axpby_normalized, left_order2_2)
{
    // TTx:         TTy:
    //    2            4
    //  o -- o       o -- o
    //  |    |       |    |
    //  4    4       4    4

    TensorTrain<double> TTx(2, 4, 2), TTy(2, 4, 4);

    randomize(TTx);
    randomize(TTy);
    left_ortho(TTx);

    check_axpby(1.0, TTx, 1.0, TTy);
    check_axpby(1.0, TTx, -2.0, TTy);
    check_axpby(3.0, TTx, 1.0, TTy);
    check_axpby(-4.0, TTx, -5.0, TTy);
}

TEST(PITTS_TensorTrain_axpby_normalized, left_order3)
{
    // TTx, TTy:
    //    3    3
    //  o -- o -- o
    //  |    |    |
    //  3    3    3

    TensorTrain<double> TTx(3, 3, 3), TTy(3, 3, 3);

    randomize(TTx);
    randomize(TTy);
    left_ortho(TTx);

    check_axpby(1.0, TTx, 1.0, TTy);
    check_axpby(1.0, TTx, -2.0, TTy);
    check_axpby(3.0, TTx, 1.0, TTy);
    check_axpby(-4.0, TTx, -5.0, TTy);
}

TEST(PITTS_TensorTrain_axpby_normalized, left_order4)
{
    // TTx, TTy:
    //    3    3    3
    //  o -- o -- o -- o
    //  |    |    |    |
    //  2    2    2    2

    TensorTrain<double> TTx(4, 2, 3), TTy(4, 2, 3);

    randomize(TTx);
    randomize(TTy);
    left_ortho(TTx);

    check_axpby(1.0, TTx, 1.0, TTy);
    check_axpby(1.0, TTx, -2.0, TTy);
    check_axpby(3.0, TTx, 1.0, TTy);
    check_axpby(-4.0, TTx, -5.0, TTy);
}

TEST(PITTS_TensorTrain_axpby_normalized, left_order5)
{
    // TTx, TTy:
    //    2    2    2    2
    //  o -- o -- o -- o -- o
    //  |    |    |    |    |
    //  2    2    2    2    2

    TensorTrain<double> TTx(5, 2, 2), TTy(5, 2, 2);

    randomize(TTx);
    randomize(TTy);
    left_ortho(TTx);

    check_axpby(1.0, TTx, 1.0, TTy);
    check_axpby(1.0, TTx, -2.0, TTy);
    check_axpby(3.0, TTx, 1.0, TTy);
    check_axpby(-4.0, TTx, -5.0, TTy);
}

TEST(PITTS_TensorTrain_axpby_normalized, left_special_cases)
{
    // TTx:                     TTy:
    //    2    3    4             1    4    2
    //  o -- o -- o -- o        o -- o -- o -- o
    //  |    |    |    |        |    |    |    |
    //  4    3    2    5        4    3    2    5

    // only testing special cases (alpha = 0, beta = 0, ...)

    const std::vector<int> dimensions = {4,3,2,5};
    TensorTrain<double> TTx(dimensions), TTy(dimensions);
    TTx.setTTranks({2,3,4});
    TTy.setTTranks({1,4,2});

    randomize(TTx);
    randomize(TTy);
    left_ortho(TTx);

    check_axpby(0.0, TTx, 0.0, TTy);

    check_axpby(0.0, TTx, 1.0, TTy);
    check_axpby(0.0, TTx, -1.0, TTy);
    check_axpby(0.0, TTx, 2.0, TTy);

    check_axpby(1.0, TTx, 0.0, TTy);
    check_axpby(-1.0, TTx, 0.0, TTy);
    check_axpby(2.0, TTx, 0.0, TTy);
}

TEST(PITTS_TensorTrain_axpby_normalized, left_highrank)
{
    // TTx:                     TTy:
    //    4    12   5             4    10   20
    //  o -- o -- o -- o        o -- o -- o -- o
    //  |    |    |    |        |    |    |    |
    //  4    3    2    5        4    3    2    5
    
    const std::vector<int> dimensions = {4,3,2,5};
    TensorTrain<double> TTx(dimensions), TTy(dimensions);
    TTx.setTTranks({4,12,5});
    TTy.setTTranks({4,10,20});

    randomize(TTx);
    randomize(TTy);
    left_ortho(TTx);

    // ranks might change after left-orthogonalization (but won't be reduced to minimal ranks)
    //printTTranks(TTx, "X");
    //printTTranks(TTy, "Y");

    check_axpby(1.0, TTx, 1.0, TTy);
    check_axpby(1.0, TTx, -2.0, TTy);
    check_axpby(3.0, TTx, 1.0, TTy);
    check_axpby(-4.0, TTx, -5.0, TTy);
}

TEST(PITTS_TensorTrain_axpby_normalized, left_lowrank)
{
    // TTx:                     TTy:
    //    1    3    2             2    3    1
    //  o -- o -- o -- o        o -- o -- o -- o
    //  |    |    |    |        |    |    |    |
    //  4    3    2    5        4    3    2    5
    
    const std::vector<int> dimensions = {4,3,2,5};
    TensorTrain<double> TTx(dimensions), TTy(dimensions);
    TTx.setTTranks({1,3,2});
    TTy.setTTranks({2,3,1});

    randomize(TTx);
    randomize(TTy);
    left_ortho(TTx);

    check_axpby(1.0, TTx, 1.0, TTy);
    check_axpby(1.0, TTx, -2.0, TTy);
    check_axpby(3.0, TTx, 1.0, TTy);
    check_axpby(-4.0, TTx, -5.0, TTy);
}

TEST(PITTS_TensorTrain_axpby_normalized, left_equal)
{
    // TTx = TTy:
    //    2    3   4
    //  o -- o -- o -- o
    //  |    |    |    |
    //  4    3    2    5

    const std::vector<int> dimensions = {4,3,2,5};
    TensorTrain<double> TTx(dimensions), TTy(dimensions);
    TTx.setTTranks({2,3,4});

    randomize(TTx);
    copy(TTx, TTy);
    left_ortho(TTx);

    // addition
    check_axpby(1.0, TTx, 1.0, TTy);
    // --> 0 tensor
    check_axpby(-1.0, TTx, 1.0, TTy);
    check_axpby(1.0, TTx, -1.0, TTy);
    // axpby
    check_axpby(2.0, TTx, -3.0, TTy);
}

TEST(PITTS_TensorTrain_axpby_normalized, left_large_tensors)
{
    // TTx:                     TTy:
    //    5    10    5             1    7    6
    //  o -- o -- o -- o        o -- o -- o -- o
    //  |    |    |    |        |    |    |    |
    //  40   30   25   20       40   30   25   20
    
    const std::vector<int> dimensions = {40,30,25,20};
    TensorTrain<double> TTx(dimensions), TTy(dimensions);
    TTx.setTTranks({5,10,5});
    TTy.setTTranks({1,7,6});

    randomize(TTx);
    randomize(TTy);
    left_ortho(TTx);

    check_axpby(1.0, TTx, 1.0, TTy, 1.e-5);
    check_axpby(1.0, TTx, -2.0, TTy, 1.e-5);
    check_axpby(3.0, TTx, 1.0, TTy, 1.e-5);
    check_axpby(-4.0, TTx, -5.0, TTy, 1.e-5);
}

TEST(PITTS_TensorTrain_axpby_normalized, left_long_tensors)
{
    // TTx:                                   TTy:
    //    4    6    7    9    8    5             3    6    9    7    6    4
    //  o -- o -- o -- o -- o -- o -- o        o -- o -- o -- o -- o -- o -- o
    //  |    |    |    |    |    |    |        |    |    |    |    |    |    |
    //  8    9    3    6    5    9    7        8    9    3    6    5    9    7
    
    const std::vector<int> dimensions = {8,9,3,6,5,9,7};
    TensorTrain<double> TTx(dimensions), TTy(dimensions);
    TTx.setTTranks({4,6,7,9,8,5});
    TTy.setTTranks({3,6,9,7,6,4});

    randomize(TTx);
    randomize(TTy);
    left_ortho(TTx);

    check_axpby(1.0, TTx, 1.0, TTy, 1.e-7);
    check_axpby(1.0, TTx, -2.0, TTy, 1.e-7);
    check_axpby(3.0, TTx, 1.0, TTy, 1.e-7);
    check_axpby(-4.0, TTx, -5.0, TTy, 1.e-7);
}


/****************************************************************************************************
 * RIGHT
 ***************************************************************************************************/

TEST(PITTS_TensorTrain_axpby_normalized, right_order1)
{
    // TTx, TTy:
    //  o
    //  |
    //  11

    TensorTrain<double> TTx(1, 11), TTy(1, 11);

    randomize(TTx);
    randomize(TTy);
    right_ortho(TTx);

    check_axpby(1.0, TTx, 1.0, TTy);
    check_axpby(1.0, TTx, -2.0, TTy);
    check_axpby(3.0, TTx, 1.0, TTy);
    check_axpby(-4.0, TTx, -5.0, TTy);
}

TEST(PITTS_TensorTrain_axpby_normalized, right_order2_0)
{
    // TTx, TTy:
    //    3
    //  o -- o
    //  |    |
    //  4    4

    TensorTrain<double> TTx(2, 4, 3), TTy(2, 4, 3);

    randomize(TTx);
    randomize(TTy);
    right_ortho(TTx);

    check_axpby(1.0, TTx, 1.0, TTy);
    check_axpby(1.0, TTx, -2.0, TTy);
    check_axpby(3.0, TTx, 1.0, TTy);
    check_axpby(-4.0, TTx, -5.0, TTy);
}

TEST(PITTS_TensorTrain_axpby_normalized, right_order2_1)
{
    // TTx:         TTy:
    //    4            2
    //  o -- o       o -- o
    //  |    |       |    |
    //  4    4       4    4

    TensorTrain<double> TTx(2, 4, 4), TTy(2, 4, 2);

    randomize(TTx);
    randomize(TTy);
    right_ortho(TTx);

    check_axpby(1.0, TTx, 1.0, TTy);
    check_axpby(1.0, TTx, -2.0, TTy);
    check_axpby(3.0, TTx, 1.0, TTy);
    check_axpby(-4.0, TTx, -5.0, TTy);
}

TEST(PITTS_TensorTrain_axpby_normalized, right_order2_2)
{
    // TTx:         TTy:
    //    2            4
    //  o -- o       o -- o
    //  |    |       |    |
    //  4    4       4    4

    TensorTrain<double> TTx(2, 4, 2), TTy(2, 4, 4);

    randomize(TTx);
    randomize(TTy);
    right_ortho(TTx);

    check_axpby(1.0, TTx, 1.0, TTy);
    check_axpby(1.0, TTx, -2.0, TTy);
    check_axpby(3.0, TTx, 1.0, TTy);
    check_axpby(-4.0, TTx, -5.0, TTy);
}

TEST(PITTS_TensorTrain_axpby_normalized, right_order3)
{
    // TTx, TTy:
    //    3    3
    //  o -- o -- o
    //  |    |    |
    //  3    3    3

    TensorTrain<double> TTx(3, 3, 3), TTy(3, 3, 3);

    randomize(TTx);
    randomize(TTy);
    right_ortho(TTx);

    check_axpby(1.0, TTx, 1.0, TTy);
    check_axpby(1.0, TTx, -2.0, TTy);
    check_axpby(3.0, TTx, 1.0, TTy);
    check_axpby(-4.0, TTx, -5.0, TTy);
}

TEST(PITTS_TensorTrain_axpby_normalized, right_order4)
{
    // TTx, TTy:
    //    3    3    3
    //  o -- o -- o -- o
    //  |    |    |    |
    //  2    2    2    2

    TensorTrain<double> TTx(4, 2, 3), TTy(4, 2, 3);

    randomize(TTx);
    randomize(TTy);
    right_ortho(TTx);

    check_axpby(1.0, TTx, 1.0, TTy);
    check_axpby(1.0, TTx, -2.0, TTy);
    check_axpby(3.0, TTx, 1.0, TTy);
    check_axpby(-4.0, TTx, -5.0, TTy);
}

TEST(PITTS_TensorTrain_axpby_normalized, right_order5)
{
    // TTx, TTy:
    //    2    2    2    2
    //  o -- o -- o -- o -- o
    //  |    |    |    |    |
    //  2    2    2    2    2

    TensorTrain<double> TTx(5, 2, 2), TTy(5, 2, 2);

    randomize(TTx);
    randomize(TTy);
    right_ortho(TTx);

    check_axpby(1.0, TTx, 1.0, TTy);
    check_axpby(1.0, TTx, -2.0, TTy);
    check_axpby(3.0, TTx, 1.0, TTy);
    check_axpby(-4.0, TTx, -5.0, TTy);
}

TEST(PITTS_TensorTrain_axpby_normalized, right_special_cases)
{
    // TTx:                     TTy:
    //    2    3    4             1    4    2
    //  o -- o -- o -- o        o -- o -- o -- o
    //  |    |    |    |        |    |    |    |
    //  4    3    2    5        4    3    2    5

    // only testing special cases (alpha = 0, beta = 0, ...)

    const std::vector<int> dimensions = {4,3,2,5};
    TensorTrain<double> TTx(dimensions), TTy(dimensions);
    TTx.setTTranks({2,3,4});
    TTy.setTTranks({1,4,2});

    randomize(TTx);
    randomize(TTy);
    right_ortho(TTx);

    check_axpby(0.0, TTx, 0.0, TTy);

    check_axpby(0.0, TTx, 1.0, TTy);
    check_axpby(0.0, TTx, -1.0, TTy);
    check_axpby(0.0, TTx, 2.0, TTy);

    check_axpby(1.0, TTx, 0.0, TTy);
    check_axpby(-1.0, TTx, 0.0, TTy);
    check_axpby(2.0, TTx, 0.0, TTy);
}

TEST(PITTS_TensorTrain_axpby_normalized, right_highrank)
{
    // TTx:                     TTy:
    //    4    12   5             4    10   20
    //  o -- o -- o -- o        o -- o -- o -- o
    //  |    |    |    |        |    |    |    |
    //  4    3    2    5        4    3    2    5
    
    const std::vector<int> dimensions = {4,3,2,5};
    TensorTrain<double> TTx(dimensions), TTy(dimensions);
    TTx.setTTranks({4,12,5});
    TTy.setTTranks({4,10,20});

    randomize(TTx);
    randomize(TTy);
    right_ortho(TTx);

    // ranks might change after right-orthogonalization (but won't be reduced to minimal ranks)
    //printTTranks(TTx, "X");
    //printTTranks(TTy, "Y");

    check_axpby(1.0, TTx, 1.0, TTy);
    check_axpby(1.0, TTx, -2.0, TTy);
    check_axpby(3.0, TTx, 1.0, TTy);
    check_axpby(-4.0, TTx, -5.0, TTy);
}

TEST(PITTS_TensorTrain_axpby_normalized, right_lowrank)
{
    // TTx:                     TTy:
    //    1    3    2             2    3    1
    //  o -- o -- o -- o        o -- o -- o -- o
    //  |    |    |    |        |    |    |    |
    //  4    3    2    5        4    3    2    5
    
    const std::vector<int> dimensions = {4,3,2,5};
    TensorTrain<double> TTx(dimensions), TTy(dimensions);
    TTx.setTTranks({1,3,2});
    TTy.setTTranks({2,3,1});

    randomize(TTx);
    randomize(TTy);
    right_ortho(TTx);

    check_axpby(1.0, TTx, 1.0, TTy);
    check_axpby(1.0, TTx, -2.0, TTy);
    check_axpby(3.0, TTx, 1.0, TTy);
    check_axpby(-4.0, TTx, -5.0, TTy);
}

TEST(PITTS_TensorTrain_axpby_normalized, right_equal)
{
    // TTx = TTy:
    //    2    3   4
    //  o -- o -- o -- o
    //  |    |    |    |
    //  4    3    2    5

    const std::vector<int> dimensions = {4,3,2,5};
    TensorTrain<double> TTx(dimensions), TTy(dimensions);
    TTx.setTTranks({2,3,4});

    randomize(TTx);
    copy(TTx, TTy);
    right_ortho(TTx);

    // addition
    check_axpby(1.0, TTx, 1.0, TTy);
    // --> 0 tensor
    check_axpby(-1.0, TTx, 1.0, TTy);
    check_axpby(1.0, TTx, -1.0, TTy);
    // axpby
    check_axpby(2.0, TTx, -3.0, TTy);
}

TEST(PITTS_TensorTrain_axpby_normalized, right_large_tensors)
{
    // TTx:                     TTy:
    //    5    10    5             1    7    6
    //  o -- o -- o -- o        o -- o -- o -- o
    //  |    |    |    |        |    |    |    |
    //  40   30   25   20       40   30   25   20
    
    const std::vector<int> dimensions = {40,30,25,20};
    TensorTrain<double> TTx(dimensions), TTy(dimensions);
    TTx.setTTranks({5,10,5});
    TTy.setTTranks({1,7,6});

    randomize(TTx);
    randomize(TTy);
    right_ortho(TTx);

    check_axpby(1.0, TTx, 1.0, TTy, 1.e-5);
    check_axpby(1.0, TTx, -2.0, TTy, 1.e-5);
    check_axpby(3.0, TTx, 1.0, TTy, 1.e-5);
    check_axpby(-4.0, TTx, -5.0, TTy, 1.e-5);
}

TEST(PITTS_TensorTrain_axpby_normalized, right_long_tensors)
{
    // TTx:                                   TTy:
    //    4    6    7    9    8    5             3    6    9    7    6    4
    //  o -- o -- o -- o -- o -- o -- o        o -- o -- o -- o -- o -- o -- o
    //  |    |    |    |    |    |    |        |    |    |    |    |    |    |
    //  8    9    3    6    5    9    7        8    9    3    6    5    9    7
    
    const std::vector<int> dimensions = {8,9,3,6,5,9,7};
    TensorTrain<double> TTx(dimensions), TTy(dimensions);
    TTx.setTTranks({4,6,7,9,8,5});
    TTy.setTTranks({3,6,9,7,6,4});

    randomize(TTx);
    randomize(TTy);
    right_ortho(TTx);

    check_axpby(1.0, TTx, 1.0, TTy, 1.e-7);
    check_axpby(1.0, TTx, -2.0, TTy, 1.e-7);
    check_axpby(3.0, TTx, 1.0, TTy, 1.e-7);
    check_axpby(-4.0, TTx, -5.0, TTy, 1.e-7);
}


/****************************************************************************************************
 * NON-1 BOUNDARY RANKS: LEFT
 ***************************************************************************************************/

TEST(PITTS_TensorTrain_axpby_normalized, left_boundaryRank_order1)
{
    // TTx, TTy:
    //   3   4
    //  -- o --
    //     |
    //     5

    TensorTrain<double> TTx(1, 5), TTy(1, 5);
    TTx.setSubTensor(0, Tensor3<double>(3,5,4));
    TTy.setSubTensor(0, Tensor3<double>(3,5,4));

    randomize(TTx);
    randomize(TTy);
    left_ortho(TTx);

    check_axpby(0.2, TTx, -0.5, TTy);
}

TEST(PITTS_TensorTrain_axpby_normalized, left_boundaryRank_order2)
{
    // TTx, TTy:
    //   2    3    3
    //  -- o -- o --
    //     |    |
    //     5    4

    TensorTrain<double> TTx({5,4}, 3), TTy({5,4}, 3);
    TTx.setSubTensor(0, Tensor3<double>(2,5,3));
    TTy.setSubTensor(0, Tensor3<double>(2,5,3));
    TTx.setSubTensor(1, Tensor3<double>(3,4,3));
    TTy.setSubTensor(1, Tensor3<double>(3,4,3));

    randomize(TTx);
    randomize(TTy);
    left_ortho(TTx);

    check_axpby(0.2, TTx, -0.5, TTy);
}

TEST(PITTS_TensorTrain_axpby_normalized, left_boundaryRank_order3)
{
    // TTx, TTy:
    //   3  5/1  3/4   3
    //  -- o -- o -- o --
    //     |    |    |
    //     5    3    2

    TensorTrain<double> TTx({5,3,2}, 0), TTy({5,3,2}, 0);
    TTx.setTTranks({5,3});
    TTy.setTTranks({1,4});
    TTx.setSubTensor(0, Tensor3<double>(3,5,5));
    TTy.setSubTensor(0, Tensor3<double>(3,5,1));
    TTx.setSubTensor(2, Tensor3<double>(3,2,3));
    TTy.setSubTensor(2, Tensor3<double>(4,2,3));

    randomize(TTx);
    randomize(TTy);
    left_ortho(TTx);

    check_axpby(0.2, TTx, -0.5, TTy);
}


TEST(PITTS_TensorTrain_axpby_normalized, left_boundaryRank_order4)
{
    // TTx, TTy:
    //   3  2/4  4/5  3/2   4
    //  -- o -- o -- o -- o --
    //     |    |    |    |
    //     2    3    4    2

    TensorTrain<double> TTx({2,3,4,2},0), TTy({2,3,4,2}, 0);
    TTx.setTTranks({2,4,3});
    TTy.setTTranks({4,5,2});
    TTx.setSubTensor(0, Tensor3<double>(3,2,2));
    TTy.setSubTensor(0, Tensor3<double>(3,2,4));
    TTx.setSubTensor(3, Tensor3<double>(3,2,4));
    TTy.setSubTensor(3, Tensor3<double>(2,2,4));

    randomize(TTx);
    randomize(TTy);
    left_ortho(TTx);

    check_axpby(0.2, TTx, -0.5, TTy);
}

TEST(PITTS_TensorTrain_axpby_normalized, left_boundaryRank_order5)
{
    // TTx, TTy:
    //   5  3/3  6/3  3/4  4/2   3
    //  -- o -- o -- o -- o -- o --
    //     |    |    |    |    |
    //     2    4    5    3    2    

    TensorTrain<double> TTx({2,4,5,3,2}, 0), TTy({2,4,5,3,2}, 0);
    TTx.setTTranks({3,6,3,4});
    TTy.setTTranks({3,3,4,2});
    TTx.setSubTensor(0, Tensor3<double>(5,2,3));
    TTy.setSubTensor(0, Tensor3<double>(5,2,3));
    TTx.setSubTensor(4, Tensor3<double>(4,2,3));
    TTy.setSubTensor(4, Tensor3<double>(2,2,3));

    randomize(TTx);
    randomize(TTy);
    left_ortho(TTx);

    check_axpby(0.2, TTx, -0.5, TTy);
}


/****************************************************************************************************
 * NON-1 BOUNDARY RANKS: RIGHT
 ***************************************************************************************************/

TEST(PITTS_TensorTrain_axpby_normalized, right_boundaryRank_order1)
{
    // TTx, TTy:
    //   3   4
    //  -- o --
    //     |
    //     5

    TensorTrain<double> TTx(1, 5), TTy(1, 5);
    TTx.setSubTensor(0, Tensor3<double>(3,5,4));
    TTy.setSubTensor(0, Tensor3<double>(3,5,4));

    randomize(TTx);
    randomize(TTy);
    right_ortho(TTx);

    check_axpby(0.2, TTx, -0.5, TTy);
}

TEST(PITTS_TensorTrain_axpby_normalized, right_boundaryRank_order2)
{
    // TTx, TTy:
    //   2    3    3
    //  -- o -- o --
    //     |    |
    //     5    4

    TensorTrain<double> TTx({5,4}, 3), TTy({5,4}, 3);
    TTx.setSubTensor(0, Tensor3<double>(2,5,3));
    TTy.setSubTensor(0, Tensor3<double>(2,5,3));
    TTx.setSubTensor(1, Tensor3<double>(3,4,3));
    TTy.setSubTensor(1, Tensor3<double>(3,4,3));

    randomize(TTx);
    randomize(TTy);
    right_ortho(TTx);

    check_axpby(0.2, TTx, -0.5, TTy);
}

TEST(PITTS_TensorTrain_axpby_normalized, right_boundaryRank_order3)
{
    // TTx, TTy:
    //   3  5/1  3/4   3
    //  -- o -- o -- o --
    //     |    |    |
    //     5    3    2

    TensorTrain<double> TTx({5,3,2}, 0), TTy({5,3,2}, 0);
    TTx.setTTranks({5,3});
    TTy.setTTranks({1,4});
    TTx.setSubTensor(0, Tensor3<double>(3,5,5));
    TTy.setSubTensor(0, Tensor3<double>(3,5,1));
    TTx.setSubTensor(2, Tensor3<double>(3,2,3));
    TTy.setSubTensor(2, Tensor3<double>(4,2,3));

    randomize(TTx);
    randomize(TTy);
    right_ortho(TTx);

    check_axpby(0.2, TTx, -0.5, TTy);
}


TEST(PITTS_TensorTrain_axpby_normalized, right_boundaryRank_order4)
{
    // TTx, TTy:
    //   3  2/4  4/5  3/2   4
    //  -- o -- o -- o -- o --
    //     |    |    |    |
    //     2    3    4    2

    TensorTrain<double> TTx({2,3,4,2},0), TTy({2,3,4,2}, 0);
    TTx.setTTranks({2,4,3});
    TTy.setTTranks({4,5,2});
    TTx.setSubTensor(0, Tensor3<double>(3,2,2));
    TTy.setSubTensor(0, Tensor3<double>(3,2,4));
    TTx.setSubTensor(3, Tensor3<double>(3,2,4));
    TTy.setSubTensor(3, Tensor3<double>(2,2,4));

    randomize(TTx);
    randomize(TTy);
    right_ortho(TTx);

    check_axpby(0.2, TTx, -0.5, TTy);
}

TEST(PITTS_TensorTrain_axpby_normalized, right_boundaryRank_order5)
{
    // TTx, TTy:
    //   5  3/3  6/3  3/4  4/2   3
    //  -- o -- o -- o -- o -- o --
    //     |    |    |    |    |
    //     2    4    5    3    2    

    TensorTrain<double> TTx({2,4,5,3,2}, 0), TTy({2,4,5,3,2}, 0);
    TTx.setTTranks({3,6,3,4});
    TTy.setTTranks({3,3,4,2});
    TTx.setSubTensor(0, Tensor3<double>(5,2,3));
    TTy.setSubTensor(0, Tensor3<double>(5,2,3));
    TTx.setSubTensor(4, Tensor3<double>(4,2,3));
    TTy.setSubTensor(4, Tensor3<double>(2,2,3));

    randomize(TTx);
    randomize(TTy);
    right_ortho(TTx);

    check_axpby(0.2, TTx, -0.5, TTy);
}
