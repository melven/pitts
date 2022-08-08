#include <gtest/gtest.h>
#include "pitts_tensortrain.hpp"
#include "pitts_tensortrain_axpby.hpp"
#include "pitts_tensortrain_axpby_imp.hpp"
#include "pitts_tensortrain_normalize.hpp"
#include "pitts_tensortrain_to_dense.hpp"
#include "eigen_test_helper.hpp"
#include "pitts_multivector_eigen_adaptor.hpp"

using namespace PITTS;
using namespace internal;

static auto check_axpby(double alpha, const TensorTrain<double>& TTx, double beta, TensorTrain<double>& TTy)
{
    const double eps = std::numeric_limits<double>::epsilon();
    TensorTrain<double> _TTy_(TTy);
    copy(TTy, _TTy_);

    double _gamma_ = _axpby_(alpha, TTx, beta, _TTy_);
    double gamma = axpby(alpha, TTx, beta, TTy);

    const double _norm_ = norm2(_TTy_);
    EXPECT_NEAR(_gamma_, gamma, std::sqrt(eps));
    EXPECT_NEAR(_norm_, 1.0, std::sqrt(eps));
    
    MultiVector<double> _y_;
    MultiVector<double> y;

    toDense(TTy, y);
    toDense(_TTy_, _y_);

    EXPECT_NEAR(ConstEigenMap(_y_), ConstEigenMap(y), std::sqrt(eps));
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
    //check_axpby(3.1, TTx, 5.7, TTy);
    //check_axpby(-1.0, TTx, 1.0, TTy);
    check_axpby(-1.0, TTx, -1.0, TTy);
    //check_axpby(-1.5, TTx, -2.0, TTy);
}


TEST(PITTS_TensorTrain_axpby_imp, random_entries)
{
    // TTx = TTy, of the form:
    //  o -- o -- o -- o -- o
    //  |    |    |    |    |
    // where all entries are random (initially), but then left-orthogonalized

    TensorTrain<double> TTx(5, 2, 2), TTy(5, 2, 2);

    const double m_max = 1.0;
   // initialize both TTs with (pseudo-)random numbers
    for (auto& core : TTx.editableSubTensors())
        for (int i2 = 0; i2 < core.r2(); i2++)
            for (int j = 0; j < core.n(); j++)
                for (int i1  = 0; i1 < core.r1(); i1++)
                    core(i1, j, i2) = (double)std::rand()/RAND_MAX*m_max;
    for (auto& core : TTy.editableSubTensors())
        for (int i2 = 0; i2 < core.r2(); i2++)
            for (int j = 0; j < core.n(); j++)
                for (int i1  = 0; i1 < core.r1(); i1++)
                    core(i1, j, i2) = (double)std::rand()/RAND_MAX*m_max;

    //quickndirty_visualizeTT(TTx);

    // left-orthogonalize both TTs
    internal::leftNormalize_range<double>(TTx, 0, TTx.dimensions().size() - 1, 0);
    internal::leftNormalize_range<double>(TTy, 0, TTy.dimensions().size() - 1, 0);

    //quickndirty_visualizeTT(TTx);

    check_axpby(1.0, TTx, 1.0, TTy);
    //check_axpby(3.1, TTx, 5.7, TTy);
}

TEST(PITTS_TensorTrain_axpby_imp, random_entries_2)
{
    // TTx = TTy, of the form:
    //  o -- o -- o -- o -- o
    //  |    |    |    |    |
    // where all entries are random (initially), but then left-orthogonalized

    TensorTrain<double> TTx(3, 3, 2), TTy(3, 3, 2);

    const double m_max = 1.0;
   // initialize both TTs with (pseudo-)random numbers
    for (auto& core : TTx.editableSubTensors())
        for (int i2 = 0; i2 < core.r2(); i2++)
            for (int j = 0; j < core.n(); j++)
                for (int i1  = 0; i1 < core.r1(); i1++)
                    core(i1, j, i2) = (double)std::rand()/RAND_MAX*m_max;
    for (auto& core : TTy.editableSubTensors())
        for (int i2 = 0; i2 < core.r2(); i2++)
            for (int j = 0; j < core.n(); j++)
                for (int i1  = 0; i1 < core.r1(); i1++)
                    core(i1, j, i2) = (double)std::rand()/RAND_MAX*m_max;

    //quickndirty_visualizeTT(TTx);

    // left-orthogonalize both TTs
    internal::leftNormalize_range<double>(TTx, 0, TTx.dimensions().size() - 1, 0);
    internal::leftNormalize_range<double>(TTy, 0, TTy.dimensions().size() - 1, 0);

    //quickndirty_visualizeTT(TTx);

    check_axpby(1.0, TTx, 1.0, TTy);
    //check_axpby(3.1, TTx, 5.7, TTy);
}

