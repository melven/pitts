#include <gtest/gtest.h>
#include "pitts_tensortrain.hpp"
#include "pitts_tensortrain_axpby.hpp"
#include "pitts_tensortrain_axpby_imp.hpp"
#include "pitts_tensortrain_to_dense.hpp"
#include "eigen_test_helper.hpp"
#include "pitts_multivector_eigen_adaptor.hpp"

using namespace PITTS;

static auto check_axpby(double alpha, const TensorTrain<double>& TTx, double beta, TensorTrain<double>& TTy)
{
    const double eps = std::numeric_limits<double>::epsilon();
    TensorTrain<double> _TTy_(TTy);
    copy(TTy, _TTy_);

    double _gamma_ = _axpby_(alpha, TTx, beta, _TTy_);
    double gamma = axpby(alpha, TTx, beta, TTy);

    const double _norm_ = norm2(_TTy_);
    EXPECT_NEAR(_norm_, 1.0, std::sqrt(eps));
    
    MultiVector<double> _y_;
    MultiVector<double> y;

    toDense(TTy, y);
    toDense(_TTy_, _y_);

    EXPECT_NEAR(ConstEigenMap(_y_), ConstEigenMap(y), 1.e-4);
}



TEST(PITTS_TensorTrain_axpby_imp, orthogonal_unit_vectors)
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

    /*
    for (auto& core : x_cores)
    {
        std::cout << "dimensions: " << core.r1() << " x " << core.n() << " x "<< core.r2() << std::endl;
        for (int j = 0; j < core.n(); j++)
        {
            for (int i1  = 0; i1 < core.r1(); i1++)
            {
                for (int i2 = 0; i2 < core.r2(); i2++)
                {
                    std::cout << core(i1, j, i2) << '\t';
                }
                std::cout << std::endl;
            }
        }
        std::cout << "\n\n\n";
    }
    */

    check_axpby(1, TTx, 1, TTy);
    //check_axpby(3, TTx, 5, TTy);
}
