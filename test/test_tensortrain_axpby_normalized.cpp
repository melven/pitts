#include <gtest/gtest.h>
#include "pitts_tensor3.hpp"
#include "pitts_tensortrain.hpp"
#include "pitts_tensortrain_axpby.hpp"
#include "pitts_tensortrain_axpby_normalized.hpp"
#include "pitts_tensortrain_normalize.hpp"
#include "pitts_tensortrain_to_dense.hpp"
#include "eigen_test_helper.hpp"
#include "pitts_multivector_eigen_adaptor.hpp"

using namespace PITTS;
using namespace internal;

/**
 * @brief Helper function for testing axpby_normalized implementation
 * 
 * @param alpha 
 * @param TTx 
 * @param beta 
 * @param TTy  
 */
static void check_axpby(double alpha, const TensorTrain<double>& TTx, double beta, TensorTrain<double>& TTy, double accuracy = std::sqrt(std::numeric_limits<double>::epsilon()))
{
    accuracy = accuracy * (abs(alpha) * norm2(TTx) + abs(beta) * norm2(TTy)); // upper bound for norm of result tensor

    TensorTrain<double> _TTy_(TTy);
    copy(TTy, _TTy_);

    const double _gamma_ = axpby_normalized(alpha, TTx, beta, _TTy_);
    const double gamma = axpby(alpha, TTx, beta, TTy);
    EXPECT_NEAR(_gamma_, gamma, accuracy);

    const double _norm_ = norm2(_TTy_);
    const double norm = norm2(TTy);
    EXPECT_NEAR(_norm_, norm, accuracy);
    
    MultiVector<double> _y_, y;
    toDense(TTy, y);
    toDense(_TTy_, _y_);
    EXPECT_NEAR(ConstEigenMap(_y_), ConstEigenMap(y), accuracy);
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
        auto& core = tt.editableSubTensors()[k];
        for (int i2 = 0; i2 < core.r2(); i2++)
            for (int j = 0; j < core.n(); j++)
                for (int i1  = 0; i1 < core.r1(); i1++)
                    core(i1, j, i2) = (double)std::rand() / RAND_MAX * max;
    }
    
}

/**
 * @brief Helper Function: Left-orthogonalizes cores in [first_idx, last_idx - 1) of tensor train.
 * 
 * @param tt            tensor train object
 * @param first_idx     first index
 * @param last_idx      last index: -1 <==> last
 */
static void ortho(TensorTrain<double>& tt, int first_idx = 0, int last_idx = -1)
{
    if (last_idx == -1) last_idx = tt.dimensions().size();
    // left-orthogonalize TT
    internal::leftNormalize_range<double>(tt, 0, last_idx - 1, 0);
}


static void printTTranks(const TensorTrain<double>& tt, const char* name)
{
    std::vector<int> ranks = tt.getTTranks();
    std::cout << "tensor train ranks of " << name << ": ";
    for (int rank : ranks) 
        std::cout << rank << "\t";
    std::cout << std::endl;
}



TEST(PITTS_TensorTrain_axpby_normalized, DISABLED_order1)
{
    // TTx, TTy:
    //  o
    //  |
    //  11

    // tensor core entries are random
    srand(1); 

    TensorTrain<double> TTx(1, 11), TTy(1, 11);

    const double m_max = 1.0;
    rand_init(TTx, m_max);
    rand_init(TTy, m_max);
    ortho(TTx);
    //ortho(TTy);

    check_axpby(1.0, TTx, 1.0, TTy);
    check_axpby(1.0, TTx, -2.0, TTy);
    check_axpby(3.0, TTx, 1.0, TTy);
    check_axpby(-4.0, TTx, -5.0, TTy);
}

TEST(PITTS_TensorTrain_axpby_normalized, order2_0)
{
    // TTx, TTy:
    //    3
    //  o -- o
    //  |    |
    //  4    4

    // tensor core entries are random
    srand(1); 

    TensorTrain<double> TTx(2, 4, 3), TTy(2, 4, 3);

    const double m_max = 1.0;
    rand_init(TTx, m_max);
    rand_init(TTy, m_max);
    ortho(TTx);
    //ortho(TTy);

    check_axpby(1.0, TTx, 1.0, TTy);
    check_axpby(1.0, TTx, -2.0, TTy);
    check_axpby(3.0, TTx, 1.0, TTy);
    check_axpby(-4.0, TTx, -5.0, TTy);
}

TEST(PITTS_TensorTrain_axpby_normalized, order2_1)
{
    // TTx:         TTy:
    //    4            2
    //  o -- o       o -- o
    //  |    |       |    |
    //  4    4       4    4

    // tensor core entries are random
    srand(1); 

    TensorTrain<double> TTx(2, 4, 4), TTy(2, 4, 2);

    const double m_max = 1.0;
    rand_init(TTx, m_max);
    rand_init(TTy, m_max);
    ortho(TTx);
    //ortho(TTy);

    check_axpby(1.0, TTx, 1.0, TTy);
    check_axpby(1.0, TTx, -2.0, TTy);
    check_axpby(3.0, TTx, 1.0, TTy);
    check_axpby(-4.0, TTx, -5.0, TTy);
}

TEST(PITTS_TensorTrain_axpby_normalized, order2_2)
{
    // TTx:         TTy:
    //    2            4
    //  o -- o       o -- o
    //  |    |       |    |
    //  4    4       4    4

    // tensor core entries are random
    srand(1); 

    TensorTrain<double> TTx(2, 4, 2), TTy(2, 4, 4);

    const double m_max = 1.0;
    rand_init(TTx, m_max);
    rand_init(TTy, m_max);
    ortho(TTx);

    check_axpby(1.0, TTx, 1.0, TTy);
    check_axpby(1.0, TTx, -2.0, TTy);
    check_axpby(3.0, TTx, 1.0, TTy);
    check_axpby(-4.0, TTx, -5.0, TTy);
}

TEST(PITTS_TensorTrain_axpby_normalized, order3)
{
    // TTx, TTy:
    //    3    3
    //  o -- o -- o
    //  |    |    |
    //  3    3    3

    // tensor core entries are random
    srand(1); 

    TensorTrain<double> TTx(3, 3, 3), TTy(3, 3, 3);

    const double m_max = 1.0;
    rand_init(TTx, m_max);
    rand_init(TTy, m_max);
    ortho(TTx);

    check_axpby(1.0, TTx, 1.0, TTy);
    check_axpby(1.0, TTx, -2.0, TTy);
    check_axpby(3.0, TTx, 1.0, TTy);
    check_axpby(-4.0, TTx, -5.0, TTy);
}

TEST(PITTS_TensorTrain_axpby_normalized, order4)
{
    // TTx, TTy:
    //    3    3    3
    //  o -- o -- o -- o
    //  |    |    |    |
    //  2    2    2    2

    // tensor core entries are random
    srand(1); 

    TensorTrain<double> TTx(4, 2, 3), TTy(4, 2, 3);

    const double m_max = 1.0;
    rand_init(TTx, m_max);
    rand_init(TTy, m_max);
    ortho(TTx);

    check_axpby(1.0, TTx, 1.0, TTy);
    check_axpby(1.0, TTx, -2.0, TTy);
    check_axpby(3.0, TTx, 1.0, TTy);
    check_axpby(-4.0, TTx, -5.0, TTy);
}

TEST(PITTS_TensorTrain_axpby_normalized, order5)
{
    // TTx, TTy:
    //    2    2    2    2
    //  o -- o -- o -- o -- o
    //  |    |    |    |    |
    //  2    2    2    2    2

    // tensor core entries are random
    srand(1); 

    TensorTrain<double> TTx(5, 2, 2), TTy(5, 2, 2);

    const double m_max = 1.0;
    rand_init(TTx, m_max);
    rand_init(TTy, m_max);
    ortho(TTx);

    check_axpby(1.0, TTx, 1.0, TTy);
    check_axpby(1.0, TTx, -2.0, TTy);
    check_axpby(3.0, TTx, 1.0, TTy);
    check_axpby(-4.0, TTx, -5.0, TTy);
}

TEST(PITTS_TensorTrain_axpby_normalized, special_cases)
{
    // TTx:                     TTy:
    //    2    3    4             1    4    2
    //  o -- o -- o -- o        o -- o -- o -- o
    //  |    |    |    |        |    |    |    |
    //  4    3    2    5        4    3    2    5

    // only testing special cases (alpha = 0, beta = 0, ...)

    // tensor core entries are random
    srand(1); 

    const std::vector<int> dimensions = {4,3,2,5};
    TensorTrain<double> TTx(dimensions), TTy(dimensions);
    TTx.setTTranks({2,3,4});
    TTy.setTTranks({1,4,2});

    const double m_max = 1.0;
    rand_init(TTx, m_max);
    rand_init(TTy, m_max);
    ortho(TTx);

    check_axpby(0.0, TTx, 0.0, TTy);

    check_axpby(0.0, TTx, 1.0, TTy);
    check_axpby(0.0, TTx, -1.0, TTy);
    check_axpby(0.0, TTx, 2.0, TTy);

    check_axpby(1.0, TTx, 0.0, TTy);
    check_axpby(-1.0, TTx, 0.0, TTy);
    check_axpby(2.0, TTx, 0.0, TTy);
}

TEST(PITTS_TensorTrain_axpby_normalized, highrank)
{
    // TTx:                     TTy:
    //    4    12   5             4    10   20
    //  o -- o -- o -- o        o -- o -- o -- o
    //  |    |    |    |        |    |    |    |
    //  4    3    2    5        4    3    2    5
    
    // tensor core entries are random
    srand(1);

    const std::vector<int> dimensions = {4,3,2,5};
    TensorTrain<double> TTx(dimensions), TTy(dimensions);
    TTx.setTTranks({4,12,5});
    TTy.setTTranks({4,10,20});

    const double m_max = 1.0;
    rand_init(TTx, m_max);
    rand_init(TTy, m_max);
    ortho(TTx);

    // ranks might change after left-orthogonalization (but won't be reduced to minimal ranks)
    //printTTranks(TTx, "X");
    //printTTranks(TTy, "Y");

    check_axpby(1.0, TTx, 1.0, TTy);
    check_axpby(1.0, TTx, -2.0, TTy);
    check_axpby(3.0, TTx, 1.0, TTy);
    check_axpby(-4.0, TTx, -5.0, TTy);
}

TEST(PITTS_TensorTrain_axpby_normalized, lowrank)
{
    // TTx:                     TTy:
    //    1    3    2             2    3    1
    //  o -- o -- o -- o        o -- o -- o -- o
    //  |    |    |    |        |    |    |    |
    //  4    3    2    5        4    3    2    5
    
    // tensor core entries are random
    srand(1);

    const std::vector<int> dimensions = {4,3,2,5};
    TensorTrain<double> TTx(dimensions), TTy(dimensions);
    TTx.setTTranks({1,3,2});
    TTy.setTTranks({2,3,1});

    const double m_max = 1.0;
    rand_init(TTx, m_max);
    rand_init(TTy, m_max);
    ortho(TTx);

    check_axpby(1.0, TTx, 1.0, TTy);
    check_axpby(1.0, TTx, -2.0, TTy);
    check_axpby(3.0, TTx, 1.0, TTy);
    check_axpby(-4.0, TTx, -5.0, TTy);
}

TEST(PITTS_TensorTrain_axpby_normalized, equal)
{
    // TTx = TTy:
    //    2    3   4
    //  o -- o -- o -- o
    //  |    |    |    |
    //  4    3    2    5

    // tensor core entries are random
    srand(1);

    const std::vector<int> dimensions = {4,3,2,5};
    TensorTrain<double> TTx(dimensions), TTy(dimensions);
    TTx.setTTranks({2,3,4});

    const double m_max = 1.0;
    rand_init(TTx, m_max);
    copy(TTx, TTy);
    ortho(TTx);

    // addition
    check_axpby(1.0, TTx, 1.0, TTy);
    // --> 0 tensor
    check_axpby(-1.0, TTx, 1.0, TTy);
    check_axpby(1.0, TTx, -1.0, TTy);
    // axpby
    check_axpby(2.0, TTx, -3.0, TTy);
}

TEST(PITTS_TensorTrain_axpby_normalized, large_tensors)
{
    // TTx:                     TTy:
    //    5    10    5             1    7    6
    //  o -- o -- o -- o        o -- o -- o -- o
    //  |    |    |    |        |    |    |    |
    //  40   30   25   20       40   30   25   20
    
    // tensor core entries are random
    srand(1);

    const std::vector<int> dimensions = {40,30,25,20};
    TensorTrain<double> TTx(dimensions), TTy(dimensions);
    TTx.setTTranks({5,10,5});
    TTy.setTTranks({1,7,6});

    const double m_max = 1.0;
    rand_init(TTx, m_max);
    rand_init(TTy, m_max);
    ortho(TTx);

    check_axpby(1.0, TTx, 1.0, TTy);
    check_axpby(1.0, TTx, -2.0, TTy);
    check_axpby(3.0, TTx, 1.0, TTy);
    check_axpby(-4.0, TTx, -5.0, TTy);
}

TEST(PITTS_TensorTrain_axpby_normalized, long_tensors)
{
    // TTx:                                   TTy:
    //    4    6    7    9    8    5             3    6    9    7    6    4
    //  o -- o -- o -- o -- o -- o -- o        o -- o -- o -- o -- o -- o -- o
    //  |    |    |    |    |    |    |        |    |    |    |    |    |    |
    //  8    9    3    6    5    9    7        8    9    3    6    5    9    7
    
    // tensor core entries are random
    srand(1);

    const std::vector<int> dimensions = {8,9,3,6,5,9,7};
    TensorTrain<double> TTx(dimensions), TTy(dimensions);
    TTx.setTTranks({4,6,7,9,8,5});
    TTy.setTTranks({3,6,9,7,6,4});

    const double m_max = 1.0;
    rand_init(TTx, m_max);
    rand_init(TTy, m_max);
    ortho(TTx);

    check_axpby(1.0, TTx, 1.0, TTy);
    check_axpby(1.0, TTx, -2.0, TTy);
    check_axpby(3.0, TTx, 1.0, TTy);
    check_axpby(-4.0, TTx, -5.0, TTy);
}


/*
// test is not very conclusive (false positive common), just keeping it in case...
TEST(PITTS_TensorTrain_axpby_normalized, unit_vectors)
{
    // TTx = TTy, of the form:
    //  o -- o -- o 
    //  |    |    |
    // with n, r = 2 for all n's and r's (except r0 = r3 = 1)
    //
    // core 0 (1 x n1 x r1)        core 1 (r1 x n2 x r2)        core 2 (r2 x n3 x 1)
    //  1 x | [ 1  0 ]                  | [ 1  0 ]                   | [ 1 ]
    //   n1 | [ 0  1 ]          r1 x n2 | [ 0  1 ]           r2 x n3 | [ 0 ]
    //        --------                  | [ 0  0 ]                   | [ 0 ]
    //           r2                     | [ 0  0 ]                   | [ 0 ]
    //                                    --------                     -----
    //                                       r2                          1
    //
    // The left-unfolded core-matrices are the canonical projection matrices diag(1,1,...)
    // ==> they are left-orthogonal

    TensorTrain<double> TTx(3, 2, 2), TTy(3, 2, 2);

    std::vector<Tensor3<double>>& x_cores = TTx.editableSubTensors();

    // 0 init needed?
    for (auto& core : x_cores)
        for (int i2 = 0; i2 < core.r2(); i2++)
            for (int j = 0; j < core.n(); j++)
                for (int i1 = 0; i1 < core.r1(); i1++)
                    core(i1, j, i2) = 0.0;

    x_cores[0](0,0,0) = 1.0;
    x_cores[0](0,1,1) = 1.0;
    x_cores[1](0,0,0) = 1.0;
    x_cores[1](1,0,1) = 1.0;
    x_cores[2](0,0,0) = 1.0;

    copy(TTx, TTy);

    check_axpby(1.0, TTx, 1.0, TTy);
    check_axpby(3.1, TTx, 5.7, TTy);
    check_axpby(-1.0, TTx, 1.0, TTy);
    check_axpby(-1.0, TTx, -1.0, TTy);
    check_axpby(-1.5, TTx, -2.0, TTy);
}
*/