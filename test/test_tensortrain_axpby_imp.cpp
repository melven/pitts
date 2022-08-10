#include <gtest/gtest.h>
#include "pitts_tensor3.hpp"
#include "pitts_tensortrain.hpp"
#include "pitts_tensortrain_axpby.hpp"
#include "pitts_tensortrain_axpby_imp.hpp"
#include "pitts_tensortrain_normalize.hpp"
#include "pitts_tensortrain_to_dense.hpp"
#include "eigen_test_helper.hpp"
#include "pitts_multivector_eigen_adaptor.hpp"

using namespace PITTS;
using namespace internal;

/**
 * @brief Helper function for testing _axpby_ implementation
 * 
 * @param alpha 
 * @param TTx 
 * @param beta 
 * @param TTy  
 */
static void check_axpby(double alpha, const TensorTrain<double>& TTx, double beta, TensorTrain<double>& TTy)
{
    const double eps = std::numeric_limits<double>::epsilon();
    TensorTrain<double> _TTy_(TTy);
    copy(TTy, _TTy_);

    const double _gamma_ = _axpby_(alpha, TTx, beta, _TTy_);
    const double gamma = axpby(alpha, TTx, beta, TTy);
    EXPECT_NEAR(_gamma_, gamma, std::sqrt(eps));

    const double _norm_ = norm2(_TTy_);
    const double norm = norm2(TTy);
    EXPECT_NEAR(_norm_, norm, std::sqrt(eps));
    
    MultiVector<double> _y_;
    MultiVector<double> y;
    toDense(TTy, y);
    toDense(_TTy_, _y_);
    EXPECT_NEAR(ConstEigenMap(_y_), ConstEigenMap(y), std::sqrt(eps));
}


/**
 * @brief Helper function: initializes a pre-existing tensor randomly and left-orthogonalizes it
 * 
 * @param tt    tensor object
 * @param max   maximal absolute value to initialize with
 */
static void rand_init_and_ortho(TensorTrain<double>& tt, double max = 1.0, int first_idx = 0, int last_idx = -1)
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
    // left-orthogonalize TT
    internal::leftNormalize_range<double>(tt, 0, last_idx - 1, 0);
}



TEST(PITTS_TensorTrain_axpby_imp, DISABLED_dim1)
{
    // TTx, TTy, of the form:
    //  o
    //  |
    // where all entries are random (initially), but then left-orthogonalized

    TensorTrain<double> TTx(1, 2, 2), TTy(1, 2, 2);
    const auto& G = TTx.subTensors()[0];

    const double m_max = 1.0;
    rand_init_and_ortho(TTx, m_max);
    rand_init_and_ortho(TTy, m_max);

    check_axpby(1.0, TTx, 1.0, TTy);
    check_axpby(-1.0, TTx, 1.0, TTy);
    check_axpby(1.0, TTx, -2.0, TTy);
    check_axpby(-4.0, TTx, -5.0, TTy);
}

TEST(PITTS_TensorTrain_axpby_imp, dim2)
{
    // TTx, TTy, of the form:
    //  o -- o
    //  |    |
    // where all entries are random (initially), but then left-orthogonalized

    TensorTrain<double> TTx(2, 4, 3), TTy(2, 4, 4);
    const auto& G = TTx.subTensors()[0];

    const double m_max = 1.0;
    rand_init_and_ortho(TTx, m_max);
    rand_init_and_ortho(TTy, m_max);

    check_axpby(1.0, TTx, 1.0, TTy);
    check_axpby(-1.0, TTx, 1.0, TTy);
    check_axpby(1.0, TTx, -2.0, TTy);
    check_axpby(-4.0, TTx, -5.0, TTy);
}

TEST(PITTS_TensorTrain_axpby_imp, random_entries)
{
    // TTx = TTy, of the form:
    //  o -- o -- o -- o -- o
    //  |    |    |    |    |
    // where all entries are random (initially), but then left-orthogonalized

    TensorTrain<double> TTx(4, 2, 2), TTy(4, 2, 2);

    const double m_max = 1.0;
    rand_init_and_ortho(TTx, m_max);
    rand_init_and_ortho(TTy, m_max);

    check_axpby(1.0, TTx, 1.0, TTy);
}

TEST(PITTS_TensorTrain_axpby_imp, random_entries_1)
{
    // TTx = TTy, of the form:
    //  o -- o -- o -- o -- o
    //  |    |    |    |    |
    // where all entries are random (initially), but then left-orthogonalized

    TensorTrain<double> TTx(5, 2, 2), TTy(5, 2, 2);

    const double m_max = 1.0;
    rand_init_and_ortho(TTx, m_max);
    rand_init_and_ortho(TTy, m_max);

    check_axpby(1.0, TTx, 1.0, TTy);
}

TEST(PITTS_TensorTrain_axpby_imp, random_entries_2)
{
    // TTx = TTy, of the form:
    //  o -- o -- o -- o
    //  |    |    |    |
    // where all entries are random (initially), but then left-orthogonalized

    TensorTrain<double> TTx(4, 3, 2), TTy(4, 3, 2);

    const double m_max = 1.0;
    rand_init_and_ortho(TTx, m_max);
    rand_init_and_ortho(TTy, m_max);

    check_axpby(1.0, TTx, 1.0, TTy);
    //check_axpby(2.0, TTx, 5.0, TTy);
}

TEST(PITTS_TensorTrain_axpby_imp, random_entries_3)
{
    // TTx = TTy, of the form:
    //  o -- o -- o
    //  |    |    |
    // where all entries are random (initially), but then left-orthogonalized

    TensorTrain<double> TTx(3, 3, 2), TTy(3, 3, 2);

    const double m_max = 1.0;
    rand_init_and_ortho(TTx, m_max);
    rand_init_and_ortho(TTy, m_max);

    //check_axpby(1.0, TTx, 1.0, TTy);
    check_axpby(2.0, TTx, 5.0, TTy);
}




TEST(PITTS_TensorTrain_axpby_imp, unit_vectors)
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


    check_axpby(1.0, TTx, 1.0, TTy); // i think it passes by pure chance here
    check_axpby(3.1, TTx, 5.7, TTy);
    check_axpby(-1.0, TTx, 1.0, TTy);
    check_axpby(-1.0, TTx, -1.0, TTy);
    check_axpby(-1.5, TTx, -2.0, TTy);
}

/*
TEST(PITTS_TensorTrain_axpby_imp, random_entries_0)
{
    // TTx = TTy, of the form:
    //  o -- o -- o -- o -- o
    //  |    |    |    |    |
    // where all entries are random (initially), but then left-orthogonalized

    TensorTrain<double> TTx(4, 2, 2), TTy(4, 2, 2);

    const double m_max = 1.0;
    rand_init_and_ortho(TTx, m_max);
    rand_init_and_ortho(TTy, m_max);

    TensorTrain<double> TTxx(5, 2, 2), TTyy(5, 2, 2);

    for (int i = 0; i < 3; i++)
    {
        const auto& xcore = TTx.subTensors()[i];
        auto& xcorenew = TTxx.editableSubTensors()[i];
        copy(xcore, xcorenew);

        const auto& ycore = TTy.subTensors()[i];
        auto& ycorenew = TTyy.editableSubTensors()[i];
        copy(ycore, ycorenew);
    }
    
    auto& xcore = TTxx.editableSubTensors()[3];
        for (int i2 = 0; i2 < xcore.r2(); i2++)
            for (int j = 0; j < xcore.n(); j++)
                for (int i1  = 0; i1 < xcore.r1(); i1++)
                    xcore(i1, j, i2) = (double)std::rand() / RAND_MAX * m_max;
    auto& xcore2 = TTxx.editableSubTensors()[4];
        for (int i2 = 0; i2 < xcore2.r2(); i2++)
            for (int j = 0; j < xcore2.n(); j++)
                for (int i1  = 0; i1 < xcore2.r1(); i1++)
                    xcore2(i1, j, i2) = (double)std::rand() / RAND_MAX * m_max;
    
    auto& ycore = TTyy.editableSubTensors()[3];
        for (int i2 = 0; i2 < ycore.r2(); i2++)
            for (int j = 0; j < ycore.n(); j++)
                for (int i1  = 0; i1 < ycore.r1(); i1++)
                    ycore(i1, j, i2) = (double)std::rand() / RAND_MAX * m_max;
    auto& ycore2 = TTyy.editableSubTensors()[4];
        for (int i2 = 0; i2 < ycore2.r2(); i2++)
            for (int j = 0; j < ycore2.n(); j++)
                for (int i1  = 0; i1 < ycore2.r1(); i1++)
                    ycore2(i1, j, i2) = (double)std::rand() / RAND_MAX * m_max;
    
    internal::leftNormalize_range<double>(TTxx, 3, TTxx.dimensions().size() - 1, 0);
    internal::leftNormalize_range<double>(TTyy, 3, TTyy.dimensions().size() - 1, 0);

    check_axpby(1.0, TTxx, 1.0, TTyy);
}
*/