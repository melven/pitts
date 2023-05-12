#include <gtest/gtest.h>
#include "pitts_tensortrain.hpp"
#include "pitts_tensortrain_random.hpp"
#include "pitts_tensortrain_axpby.hpp"
#include "pitts_tensortrain_axpby_normalized.hpp"
#include "pitts_tensortrain_dot.hpp"
#include "pitts_tensortrain_norm.hpp"
#include "pitts_tensortrain_normalize.hpp"

using namespace PITTS;

static double pow2(const double x)
{
    return x*x;
}

static auto check_axpby(double alpha, const TensorTrain<double>& TTx, double beta, TensorTrain<double>& TTy, double accuracy = 10*std::sqrt(std::numeric_limits<double>::epsilon()))
{
    TensorTrain<double> TTz(TTy);
    const double gamma = axpby(alpha, TTx, beta, TTz, accuracy);
    const double TTz_norm = norm2(TTz);
    const double TTx_norm = norm2(TTx);
    const double TTy_norm = norm2(TTy);

    const double TTresult_norm = std::sqrt(std::max(0.0, pow2(alpha*TTx_norm)+pow2(beta*TTy_norm) + 2*alpha*beta*dot(TTx,TTy)));
    const double absolute_accuracy = accuracy * (TTx_norm + TTy_norm + TTresult_norm); // add up all three norms to account for error in TTresult_norm calculation
    const double& relative_accuracy = accuracy;

    // check gamma (against dot product)
    EXPECT_NEAR(gamma, TTresult_norm, absolute_accuracy);

    if (alpha != 0.0 || beta != 0.0) // if alpha = beta = 0, no claims for entries of TTz are made
    {
        // check ||TTz|| == 1.0
        EXPECT_NEAR(1.0, norm2(TTz), relative_accuracy);
        
        // check TTz is orthogonal and the flag agrees with it
        EXPECT_TRUE(internal::is_normalized(TTz, TTz.isOrthogonal(), relative_accuracy));
    }

    // check TTz is correct
    const auto nDim = TTz.dimensions().size();
    TensorTrain<double> testTT(TTz); // hope that boundary ranks of TTx, TTy, TTz are all the same
    if( nDim == 1 && testTT.subTensor(0).r1() == 1 && testTT.subTensor(nDim-1).r2() == 1 )
    {
        for(int i = 0; i < TTz.dimensions()[0]; i++)
        {
            testTT.setUnit({i});
            EXPECT_NEAR(alpha*dot(TTx,testTT)+beta*dot(TTy,testTT), gamma*dot(TTz,testTT), absolute_accuracy);
        }
    }
    else if( nDim == 2 && testTT.subTensor(0).r1() == 1 && testTT.subTensor(nDim-1).r2() == 1 )
    {
        for(int i = 0; i < TTz.dimensions()[0]; i++)
            for(int j = 0; j < TTz.dimensions()[1]; j++)
            {
                testTT.setUnit({i,j});
                EXPECT_NEAR(alpha * dot(TTx, testTT) + beta * dot(TTy, testTT), gamma * dot(TTz, testTT), absolute_accuracy);
            }
    }
    else
    {
        for(int i = 0; i < 10; i++)
        {
            randomize(testTT);
            EXPECT_NEAR(alpha * dot(TTx, testTT) + beta * dot(TTy, testTT), gamma * dot(TTz, testTT), absolute_accuracy);
        }
    }

    return std::make_pair(gamma, TTz);
}


/***
 * uebernommen von test_tensortrain_axbpy_normalized.cpp
****/

static void printTTranks(const TensorTrain<double>& tt, const char* name)
{
    std::vector<int> ranks = tt.getTTranks();
    std::cout << "tensor train ranks of " << name << ": ";
    for (int rank : ranks) 
        std::cout << rank << "\t";
    std::cout << std::endl;
}


/****************************************************************************************************
 * X NORMALIZED
 ***************************************************************************************************/

TEST(PITTS_TensorTrain_axpby, x_special_cases)
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
    leftNormalize(TTx, 0.);

    check_axpby(0.0, TTx, 0.0, TTy);

    check_axpby(0.0, TTx, 1.0, TTy);
    check_axpby(0.0, TTx, -1.0, TTy);
    check_axpby(0.0, TTx, 2.0, TTy);

    check_axpby(1.0, TTx, 0.0, TTy);
    check_axpby(-1.0, TTx, 0.0, TTy);
    check_axpby(2.0, TTx, 0.0, TTy);
}

TEST(PITTS_TensorTrain_axpby, x_equal)
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
    rightNormalize(TTx, 0.);

    // addition
    check_axpby(1.0, TTx, 1.0, TTy);
    // --> 0 tensor
    check_axpby(-1.0, TTx, 1.0, TTy);
    check_axpby(1.0, TTx, -1.0, TTy);
    // axpby
    check_axpby(2.0, TTx, -3.0, TTy);
}

TEST(PITTS_TensorTrain_axpby, x_large_tensors)
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
    leftNormalize(TTx, 0.);

    check_axpby(0.2, TTx, -0.5, TTy);
}

TEST(PITTS_TensorTrain_axpby, x_long_tensors)
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
    rightNormalize(TTx, 0.);

    check_axpby(0.2, TTx, -0.5, TTy);
}

TEST(PITTS_TensorTrain_axpby, x_boundaryRanks)
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
    leftNormalize(TTx, 0.);

    check_axpby(0.2, TTx, -0.5, TTy);
}


/****************************************************************************************************
 * Y NORMALIZED
 ***************************************************************************************************/

TEST(PITTS_TensorTrain_axpby, y_special_cases)
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
    rightNormalize(TTy, 0.);

    check_axpby(0.0, TTx, 0.0, TTy);

    check_axpby(0.0, TTx, 1.0, TTy);
    check_axpby(0.0, TTx, -1.0, TTy);
    check_axpby(0.0, TTx, 2.0, TTy);

    check_axpby(1.0, TTx, 0.0, TTy);
    check_axpby(-1.0, TTx, 0.0, TTy);
    check_axpby(2.0, TTx, 0.0, TTy);
}

TEST(PITTS_TensorTrain_axpby, y_equal)
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
    leftNormalize(TTy, 0.);

    // addition
    check_axpby(1.0, TTx, 1.0, TTy);
    // --> 0 tensor
    check_axpby(-1.0, TTx, 1.0, TTy);
    check_axpby(1.0, TTx, -1.0, TTy);
    // axpby
    check_axpby(2.0, TTx, -3.0, TTy);
}

TEST(PITTS_TensorTrain_axpby, y_large_tensors)
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
    rightNormalize(TTy, 0.);

    check_axpby(0.2, TTx, -0.5, TTy);
}

TEST(PITTS_TensorTrain_axpby, y_long_tensors)
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
    leftNormalize(TTy, 0.);

    check_axpby(0.2, TTx, -0.5, TTy);
}

TEST(PITTS_TensorTrain_axpby, y_boundaryRanks)
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
    rightNormalize(TTy, 0.);

    check_axpby(0.2, TTx, -0.5, TTy);
}


/****************************************************************************************************
 * NONE
 ***************************************************************************************************/

TEST(PITTS_TensorTrain_axpby, none_special_cases)
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

    check_axpby(0.0, TTx, 0.0, TTy);

    check_axpby(0.0, TTx, 1.0, TTy);
    check_axpby(0.0, TTx, -1.0, TTy);
    check_axpby(0.0, TTx, 2.0, TTy);

    check_axpby(1.0, TTx, 0.0, TTy);
    check_axpby(-1.0, TTx, 0.0, TTy);
    check_axpby(2.0, TTx, 0.0, TTy);
}

TEST(PITTS_TensorTrain_axpby, none_equal)
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

    // addition
    check_axpby(1.0, TTx, 1.0, TTy, 1.e-7);
    // --> 0 tensor
    check_axpby(-1.0, TTx, 1.0, TTy, 1.e-7);
    check_axpby(1.0, TTx, -1.0, TTy, 1.e-7);
    // axpby
    check_axpby(2.0, TTx, -3.0, TTy, 1.e-7);
}

TEST(PITTS_TensorTrain_axpby, none_large_tensors)
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

    check_axpby(0.2, TTx, -0.5, TTy);
}

TEST(PITTS_TensorTrain_axpby, none_long_tensors)
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

    check_axpby(0.2, TTx, -0.5, TTy);
}

TEST(PITTS_TensorTrain_axpby, none_boundaryRanks)
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

    check_axpby(0.2, TTx, -0.5, TTy);
}


/****************************************************************************************************
 * BOTH NORMALIZED
 ***************************************************************************************************/

TEST(PITTS_TensorTrain_axpby, both_special_cases)
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
    rightNormalize(TTx, 0.);
    leftNormalize(TTy, 0.);

    check_axpby(0.0, TTx, 0.0, TTy);

    check_axpby(0.0, TTx, 1.0, TTy);
    check_axpby(0.0, TTx, -1.0, TTy);
    check_axpby(0.0, TTx, 2.0, TTy);

    check_axpby(1.0, TTx, 0.0, TTy);
    check_axpby(-1.0, TTx, 0.0, TTy);
    check_axpby(2.0, TTx, 0.0, TTy);
}

TEST(PITTS_TensorTrain_axpby, both_equal)
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
    rightNormalize(TTx, 0.);
    copy(TTx, TTy);

    // addition
    check_axpby(1.0, TTx, 1.0, TTy);
    // --> 0 tensor
    check_axpby(-1.0, TTx, 1.0, TTy);
    check_axpby(1.0, TTx, -1.0, TTy);
    // axpby
    check_axpby(2.0, TTx, -3.0, TTy);
}

TEST(PITTS_TensorTrain_axpby, both_large_tensors)
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
    leftNormalize(TTx, 0.);
    rightNormalize(TTy, 0.);

    check_axpby(0.2, TTx, -0.5, TTy);
}

TEST(PITTS_TensorTrain_axpby, both_long_tensors)
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
    leftNormalize(TTx, 0.);
    leftNormalize(TTy, 0.);

    check_axpby(0.2, TTx, -0.5, TTy);
}

TEST(PITTS_TensorTrain_axpby, both_boundaryRanks)
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
    leftNormalize(TTx, 0.);
    rightNormalize(TTy, 0.);

    check_axpby(0.2, TTx, -0.5, TTy);
}
