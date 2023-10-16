// Copyright (c) 2022 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Manuel Joey Becklas
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_tensortrain_axpby_normalized_impl.hpp
* @brief addition for simple tensor train format where one of the tensors is normalized
* @author Manuel Joey Becklas <Manuel.Becklas@DLR.de>
* @date 2022-09-06
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_AXPBY_NORMALIZED_IMPL_HPP
#define PITTS_TENSORTRAIN_AXPBY_NORMALIZED_IMPL_HPP

//#define VERBOSE

// includes
#include <iostream>
#include <iomanip>
#include <vector>
#include <cassert>
#include "pitts_tensortrain_axpby_normalized.hpp"
#include "pitts_eigen.hpp"
#include "pitts_tensor2_eigen_adaptor.hpp"
#include "pitts_tensortrain.hpp"
#include "pitts_tensortrain_norm.hpp"
#include "pitts_tensortrain_normalize.hpp"
#include "pitts_tensor2.hpp"
#include "pitts_tensor3_split.hpp"
#include "pitts_performance.hpp"
#include "pitts_chunk_ops.hpp"
#include "pitts_tensor3_unfold.hpp"
#include "pitts_timer.hpp"
#ifdef PITTS_DIRECT_MKL_GEMM
#include <mkl_cblas.h>
#endif


namespace PITTS
{
    namespace internal
    {
#ifdef PITTS_DIRECT_MKL_GEMM
        inline void cblas_gemm_mapper2(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB, const CBLAS_INDEX M, const CBLAS_INDEX N, const CBLAS_INDEX K, const double alpha, const double * A, const CBLAS_INDEX lda, const double * B, const CBLAS_INDEX ldb, const double beta, double * C, const CBLAS_INDEX ldc)
        {
        cblas_dgemm(layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        }

        inline void cblas_gemm_mapper2(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB, const CBLAS_INDEX M, const CBLAS_INDEX N, const CBLAS_INDEX K, const float alpha, const float * A, const CBLAS_INDEX lda, const float * B, const CBLAS_INDEX ldb, const float beta, float * C, const CBLAS_INDEX ldc)
        {
        cblas_sgemm(layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        }
#endif
        //! contraction (normal GEMM) for axpby_leftOrthogonalized: Mmt <- Mx^adj * Mytu
        template<typename T>
        void axpby_ortho_contract1l(const ConstTensor2View<T>& Mx, const ConstTensor2View<T>& Mytu, Tensor2<T>& Mmt)
        {
            const auto n = Mx.r2();
            const auto m = Mytu.r2();
            const auto k = Mytu.r1();
            const auto timer = PITTS::performance::createScopedTimer<Tensor3<T>>(
                {{"n", "m", "k"}, {n, m, k}},                                     // arguments
                {{n*m*k*kernel_info::FMA<T>()},                                     // flops
                 {(n*k+m*k)*kernel_info::Load<T>() + n*m*kernel_info::Store<T>()}} // data
            );

            Mmt.resize(n, m);

            auto mapMx = ConstEigenMap(Mx);
            auto mapMytu = ConstEigenMap(Mytu);
            auto mapMmt = EigenMap(Mmt);
#ifndef PITTS_DIRECT_MKL_GEMM
            mapMmt.noalias() = mapMx.adjoint() * mapMytu;
#else
            cblas_gemm_mapper2(CblasColMajor, CblasTrans, CblasNoTrans, mapMmt.rows(), mapMmt.cols(), mapMx.rows(), T(1), mapMx.data(), mapMx.colStride(), mapMytu.data(), mapMytu.colStride(), T(0), mapMmt.data(), mapMmt.colStride());
#endif
        }

        //! contraction (normal GEMM) for axpby_leftOrthogonalized: Mmt <- Mytl * Mx^adj
        template<typename T>
        void axpby_ortho_contract1r(const ConstTensor2View<T>& Mytl, const ConstTensor2View<T>& Mx, Tensor2<T>& Mmt)
        {
            const auto n = Mytl.r1();
            const auto m = Mx.r1();
            const auto k = Mytl.r2();
            const auto timer = PITTS::performance::createScopedTimer<Tensor3<T>>(
                {{"n", "m", "k"}, {n, m, k}},                                     // arguments
                {{n*m*k*kernel_info::FMA<T>()},                                     // flops
                 {(n*k+m*k)*kernel_info::Load<T>() + n*m*kernel_info::Store<T>()}} // data
            );

            Mmt.resize(n, m);

            auto mapMytl = ConstEigenMap(Mytl);
            auto mapMx = ConstEigenMap(Mx);
            auto mapMmt = EigenMap(Mmt);
#ifndef PITTS_DIRECT_MKL_GEMM
            mapMmt.noalias() = mapMytl * mapMx.adjoint();
#else
            cblas_gemm_mapper2(CblasColMajor, CblasNoTrans, CblasTrans, mapMmt.rows(), mapMmt.cols(), mapMytl.cols(), T(1), mapMytl.data(), mapMytl.colStride(), mapMx.data(), mapMx.colStride(), T(0), mapMmt.data(), mapMmt.colStride());
#endif
        }

        //! contraction (normal GEMM) for axpby_*Orthogonalized: Mytu <- Mytu - Mx * Mmt OR Mytl <- Mytl - Mmt * Mx
        template<typename T>
        void axpby_ortho_contract2(const ConstTensor2View<T>& Mx, const ConstTensor2View<T>& Mmt, Tensor2View<T>& Mytu)
        {
            const auto n = Mx.r1();
            const auto m = Mmt.r2();
            const auto k = Mx.r2();
            const auto timer = PITTS::performance::createScopedTimer<Tensor3<T>>(
                {{"n", "m", "k"}, {n, m, k}},                                     // arguments
                {{n*m*k*kernel_info::FMA<T>()},                                     // flops
                 {(n*k+m*k)*kernel_info::Load<T>() + n*m*kernel_info::Store<T>()}} // data
            );

            auto mapMx = ConstEigenMap(Mx);
            auto mapMmt = ConstEigenMap(Mmt);
            auto mapMytu = EigenMap(Mytu);
#ifndef PITTS_DIRECT_MKL_GEMM
            mapMytu -= mapMx * mapMmt;
#else
            cblas_gemm_mapper2(CblasColMajor, CblasNoTrans, CblasNoTrans, mapMytu.rows(), mapMytu.cols(), mapMx.cols(), T(-1), mapMx.data(), mapMx.colStride(), mapMmt.data(), mapMmt.colStride(), T(1), mapMytu.data(), mapMytu.colStride());
#endif
        }

        //! contraction (normal GEMM) for axpby_*Orthogonalized: C <- A * B
        template<typename T>
        void axpby_ortho_contract3(const ConstTensor2View<T>& A, const ConstTensor2View<T>& B, Tensor2View<T> C)
        {
            const auto n = A.r1();
            const auto m = B.r2();
            const auto k = A.r2();
            const auto timer = PITTS::performance::createScopedTimer<Tensor3<T>>(
                {{"n", "m", "k"}, {n, m, k}},                                     // arguments
                {{n*m*k*kernel_info::FMA<T>()},                                     // flops
                 {(n*k+m*k)*kernel_info::Load<T>() + n*m*kernel_info::Store<T>()}} // data
            );

            auto mapA = ConstEigenMap(A);
            auto mapB = ConstEigenMap(B);
            auto mapC = EigenMap(C);
#ifndef PITTS_DIRECT_MKL_GEMM
            mapC.noalias() = mapA * mapB;
#else
            cblas_gemm_mapper2(CblasColMajor, CblasNoTrans, CblasNoTrans, mapC.rows(), mapC.cols(), mapA.cols(), T(1), mapA.data(), mapA.colStride(), mapB.data(), mapB.colStride(), T(0), mapC.data(), mapC.colStride());
#endif
        }

        /**
         * @brief Componentwise axpy for Tensor3 objects.
         * y <- a*x + y
         * 
         * @tparam T    underlying data type
         * @param a     scalar a
         * @param x     [in] Tensor3 x
         * @param y     [in,out] Tensor3 y
         */
        template<typename T>
        void t3_axpy(const T a, const Tensor3<T>& x, Tensor3<T>& y)
        {
            assert(x.r1() == y.r1());
            assert(x.n()  == y.n());
            assert(x.r2() == y.r2());

            const int r1 = x.r1();
            const int n = x.n();
            const int r2 = x.r2();

            const auto timer = PITTS::performance::createScopedTimer<Tensor3<T>>(
                {{"r1", "n", "r2"}, {r1, n, r2}},                                     // arguments
                {{r1*n*r2*kernel_info::FMA<T>()},                                     // flops
                 {r1*n*r2*kernel_info::Load<T>() + r1*n*r2*kernel_info::Update<T>()}} // data: load x ; update y
            );
            
#pragma omp parallel for collapse(3) schedule(static) if(r1*n*r2 > 500)
            for(int i2 = 0; i2 < r2; i2++)
                for(int j = 0; j < n; j++)
                    for(int i1 = 0; i1 < r1; i1++)
                        y(i1,j,i2) += a * x(i1,j,i2);
        }


        /**
         * @brief Compute C <- concat(Up, Lo, dim=1), the concationation of Up and Lo along the first dimension.
         * 
         * @tparam T 
         * @param Up [in]
         * @param Lo [in]
         * @param C  [out]
         */
        template <typename T>
        inline void t3_concat1(const Tensor3<T>& Up, const Tensor3<T>& Lo, Tensor3<T>& C)
        {
            const int r1u    = Up.r1();
            const int r1l    = Lo.r1();
            const int n      = Up.n();
            const int r2     = Up.r2();

            assert(n  == Lo.n());
            assert(r2 == Lo.r2());

            const auto timer = PITTS::performance::createScopedTimer<Tensor3<T>>(
                {{"r1upp", "r1low", "n", "r2"}, {r1u, r1l, n, r2}},                                       // arguments
                {{(r1u*n*r2 + r1l*n*r2)*kernel_info::NoOp<T>()},                                          // flops
                 {(r1u*n*r2 + r1l*n*r2)*kernel_info::Load<T>() + (r1u+r1l)*n*r2*kernel_info::Store<T>()}} // data: load Up,Lo ; store C
            );

            C.resize(r1u + r1l, n, r2);
            
#pragma omp parallel for schedule(static) collapse(2)
            for (int i2 = 0; i2 < r2; i2++)
                for (int j = 0; j < n; j++)
                {
                    for (int i1 = 0; i1 < r1u; i1++)
                    {
                        C(i1, j, i2) = Up(i1, j, i2);
                    }
                    for (int i1 = 0; i1 < r1l; i1++)
                    {
                        C(i1 + r1u, j, i2) = Lo(i1, j, i2);
                    }
                }
        }

        
        /**
         * @brief Compute C <- concat(Le, Ri, dim=3), 
         * the concatination of Le and Ri along the third dimension.
         * 
         * @tparam T 
         * @param Le [in] Tensor3
         * @param Ri [in] Tensor2
         * @param C [out] Tensor3
         */
        template <typename T>
        inline void t3_concat3(const Tensor3<T>& Le, const Tensor3<T>& Ri, Tensor3<T>& C)
        {
            const int r1     = Le.r1();
            const int n      = Le.n();
            const int r2l    = Le.r2();
            const int r2r    = Ri.r2();

            assert(r1 == Ri.r1());
            assert(n  == Ri.n());

            const auto timer = PITTS::performance::createScopedTimer<Tensor3<T>>(
                {{"r1", "n", "r2left", "r2right"}, {r1, n, r2l, r2r}},                                    // arguments
                {{(r1*n*r2l + r1*n*r2r)*kernel_info::NoOp<T>()},                                          // flops
                 {(r1*n*r2l + r1*n*r2r)*kernel_info::Load<T>() + r1*n*(r2l+r2r)*kernel_info::Store<T>()}} // data: load Le,Ri ; store C
            );

            C.resize(r1, n, r2l + r2r);

#pragma omp parallel
{
#pragma omp for schedule(static) collapse(2) nowait
            for (int i2 = 0; i2 < r2l; i2++)
                for(int j = 0; j < n; j++)
                {
                    for (int i1 = 0; i1 < r1; i1++)
                    {
                        C(i1, j, i2) = Le(i1, j, i2);
                    }
                }
#pragma omp for schedule(static) collapse(2) nowait
            for (int i2 = 0; i2 < r2r; i2++)
                for(int j = 0; j < n; j++)
                {
                    for (int i1 = 0; i1 < r1; i1++)
                    {
                        C(i1, j, i2 + r2l) = Ri(i1, j, i2);
                    }
                }
}
        }


        /**
         * @brief Compute Tz <- ( (Tx;0) , Q )
         * the concatination of Tx, Q (interpreted as left-unfolded 3Tensor), filled with 0's as follows:
         * | Tx  \ |
         * | \   Q |
         * | 0   \ |
         * 
         * @tparam T 
         * @param Tx    [in]  Tensor3
         * @param Q     [in]  Tensor2
         * @param Tz    [out] Tensor3
         */
        template <typename T>
        inline void concat3_X0Q(const Tensor3<T>& Tx, const Tensor2<T>& Q, Tensor3<T>& Tz)
        {
            const int n = Tx.n();
            const int Xr1 = Tx.r1();
            const int Xr2 = Tx.r2();
            const int Qr1 = Q.r1() / n;
            const int Qr2 = Q.r2();
            const int r1 = Qr1;
            const int r2 = Xr2 + Qr2;
            assert(Q.r1() % n == 0);
            assert(Qr1 >= Xr1);

            const auto timer = PITTS::performance::createScopedTimer<Tensor3<T>>(
                {{"n(all)", "Xr1", "Xr2", "Qr1", "Qr2"}, {n, Xr1, Xr2, Qr1, Qr2}},                   // arguments
                {{r1*n*r2*kernel_info::NoOp<T>()},                                                   // flops
                 {(Xr1*n*Xr2 + Qr1*n*Qr2)*kernel_info::Load<T>() + r1*n*r2*kernel_info::Store<T>()}} // data: load Tx,Q ; store Tz
            );

            Tz.resize(r1, n, r2);

#pragma omp parallel
{
#pragma omp for schedule(static) collapse(2) nowait
            for (int i2 = 0; i2 < Xr2; i2++)
            {
                for (int j = 0; j < n; j++)
                {
                    int i1;
                    for (i1 = 0; i1 < Xr1; i1++)
                    {
                        Tz(i1, j, i2) = Tx(i1, j, i2);
                    }
                    for (i1; i1 < r1; i1++)
                    {
                        Tz(i1, j, i2) = 0;
                    }
                }
            }
#pragma omp for schedule(static) collapse(2) nowait
            for (int i2 = 0; i2 < Qr2; i2++)
            {
                for (int j = 0; j < n; j++)
                {
                    for (int i1 = 0; i1 < Qr1; i1++)
                    {
                        Tz(i1, j, i2+Xr2) = Q(i1 + j*Qr1 , i2);
                    }
                }
            }
} // end omp parallel
        }


        /**
         * @brief Compute Tz <- ( (Tx,0) ; Q )
         * the concatination of Tx, Q (interpreted as right-unfolded 3Tensor), filled with 0's as follows:
         * | Tx  0 |
         * | - Q - |
         * 
         * @tparam T 
         * @param Tx    [in]  Tensor3
         * @param Q     [in]  Tensor2
         * @param Tz    [out] Tensor3
         */
        template <typename T>
        inline void concat1_X0Q(const Tensor3<T>& Tx, const Tensor2<T>& Q, Tensor3<T>& Tz)
        {
            const int n = Tx.n();
            const int Xr1 = Tx.r1();
            const int Xr2 = Tx.r2();
            const int Qr1 = Q.r1();
            const int Qr2 = Q.r2() / n;
            const int r1 = Xr1 + Qr1;
            const int r2 = Qr2;
            assert(Q.r2() % n == 0);
            assert(Qr2 >= Xr2);

            const auto timer = PITTS::performance::createScopedTimer<Tensor3<T>>(
                {{"n(all)", "Xr1", "Xr2", "Qr1", "Qr2"}, {n, Xr1, Xr2, Qr1, Qr2}},                   // arguments
                {{r1*n*r2*kernel_info::NoOp<T>()},                                                   // flops
                 {(Xr1*n*Xr2 + Qr1*n*Qr2)*kernel_info::Load<T>() + r1*n*r2*kernel_info::Store<T>()}} // data: load Tx,Q ; store Tz
            );

            Tz.resize(r1, n, r2);
            
#pragma omp parallel
{
#pragma omp for schedule(static) collapse(2) nowait
            for (int i2 = 0; i2 < Xr2; i2++)
            {
                for (int j = 0; j < n; j++)
                {
                    for (int i1 = 0; i1 < Xr1; i1++)
                    {
                        Tz(i1, j, i2) = Tx(i1, j, i2);
                    }
                    for (int i1 = 0; i1 < Qr1; i1++)
                    {
                        Tz(i1+Xr1, j, i2) = Q(i1, j + i2*n);
                    }
                }
            }
#pragma omp for schedule(static) collapse(2) nowait
            for (int i2 = Xr2; i2 < r2; i2++)
            {
                for (int j = 0; j < n; j++)
                {
                    for (int i1 = 0; i1 < Xr1; i1++)
                    {
                        Tz(i1, j, i2) = 0;
                    }
                    for (int i1 = 0; i1 < Qr1; i1++)
                    {
                        Tz(i1+Xr1, j, i2) = Q(i1, j + i2*n);
                    }
                }
            }
} // end omp parallel
        }


        // implement TT is_normalized
        template<typename T>
        bool is_normalized(const TensorTrain<T>& tt, TT_Orthogonality orthog, double eps)
        {
            if (orthog == TT_Orthogonality::none)
                return false;

            using EigenMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
            
            const auto checkOrthogonalColumns = [orthog, eps](int iDim, const EigenMatrix& Q)
            {
                const EigenMatrix orthErr = Q.transpose() * Q - EigenMatrix::Identity(Q.cols(), Q.cols());
                if (orthErr.array().abs().maxCoeff() > eps)
                {
                    std::cout << "Error: Sub-Tensor " << iDim << " should be " << (orthog == TT_Orthogonality::left ? "left" : "right") << "-orthogonal I-V^TV is:\n";
                    std::cout << orthErr << "\n";
                    std::cout << "And the tolerance is: " << eps << "\n";
                    return false;
                }
                return true;
            };

            const int nDim = tt.dimensions().size();

            if( orthog == TT_Orthogonality::left )
            {
                for(int iDim = 0; iDim+1 < nDim; iDim++)
                {
                    const auto subT = ConstEigenMap(unfold_left(tt.subTensor(iDim)));
                    if( !checkOrthogonalColumns(iDim, subT) )
                        return false;
                }
            }
            else // orthog == TT_Orthogonality::right
            {
                for(int iDim = nDim-1; iDim > 0; iDim--)
                {
                    const auto subT = ConstEigenMap(unfold_right(tt.subTensor(iDim)));
                    if( !checkOrthogonalColumns(iDim, subT.transpose()) )
                        return false;
                }
            }

            return true;
        }

        
        // implement TT axpby_leftOrtho
        template <typename T>
        void axpby_leftOrthogonalize(T alpha, const TensorTrain<T>& TTx_ortho, T beta, TensorTrain<T>& TTy)
        {
            const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();

            const auto& TTx = TTx_ortho;
            const int d = TTx.dimensions().size(); // order d
            std::vector<Tensor3<T>> Tdummy(2); // dummy 3Tensor to update TTy with: {Tdummy[0], Tdummy[1]} = {Tz, dimension_dummy}

            // scale last tensor cores
            Tensor3<T> x_last_core;
            copy(TTx.subTensor(d-1), x_last_core);
            internal::t3_scale(alpha, x_last_core);
            copy(TTy.subTensor(d-1), Tdummy[0]); // this copy can be avoided with direct access to TTy.subTensor
            internal::t3_scale(beta, Tdummy[0]);
            if (d == 1)
            {
                internal::t3_axpy((T)1, x_last_core, Tdummy[0]);
                TTy.setSubTensor(d-1, std::move(Tdummy[0]));
                return;
            }
            Tdummy[0] = TTy.setSubTensor(d-1, std::move(Tdummy[0]));

            Tensor3<T> Tytu;        // upper half of Y tilde
            Tensor3<T> Tytl;        // lower half of Y tilde
            Tensor2<T> Mmt;     // Mmt = Mx^adj * Mytu (= Mxt^adj * Myt), where adj is the adjoint matrix (conjugate transpose)
            Tensor3<T> Ttmp;        // short-lived 3Tensor: to take QR decomposition of
            
            // initialize Tytu, Tytl for k == 0
            copy(TTy.subTensor(0), Tytu); // can avoid this copy too with extra logic in first iteration
            Tytl.resize(0, TTy.subTensor(0).n(), TTy.subTensor(0).r2());

            for (int k = 0; k < d - 1; k++)
            {
                //
                // Note: If Mxt is square matrix, most of the calculation (especially QR) can be skipped
                //

                // convenience references
                const Tensor3<T>& Ty1 = TTy.subTensor(k+1);
                const Tensor3<T>& Tx =  TTx.subTensor(k);
                // convenience variables for unfolded tensors
                ConstTensor2View<T> Mx = unfold_left(Tx);
                Tensor2View<T> Mytu = unfold_left(Tytu);

                // Mmt <- Mx^adj * Mytu
                axpby_ortho_contract1l<T>(Mx, Mytu, Mmt);
                
                // Mytu <- Mytu - Mx * Mmt
                axpby_ortho_contract2(Mx, Mmt, Mytu);
                
                // Optimization potential:
                // can save the below copy of Tytu if we directly store the above result into Ttmp
                // but for concat1 it can't be done very smoothly (x0x0x0 pattern within col)
                // (possible to store xxx000 and do permutation on Q (Q right?) afterwards
                // within the concat3_X0Q function)
                // for rightOrtho we do concat3 and don't have said problem (just need to do 
                // EigenMap by hand with wanted dimensions)

                // Ttmp <- concat(Tytu, Tytl, dim=1)
                internal::t3_concat1(Tytu, Tytl, Ttmp);

                // [Q, R] <- QR(Mtmp)
                const int r1 = Tytu.r1() + Tytl.r1(); // r_{k-1} + st_{k-1}
                const int n_k = Tx.n(); // n_k
                const int r2 = Tx.r2(); // r_k
                assert(r1*n_k - r2 >= 0);
                const auto& [Q, R] = internal::normalize_qb(unfold_left(Ttmp), true, T(0), r1*n_k - r2, true);
                
                // calculate Tz: Tdummy[0] <- ((Tx;0), Q)
                internal::concat3_X0Q(Tx, Q, Tdummy[0]);

                // calculate upper half of new Tyt: Tytu <- Mmt *1 Ty1 (mode-1 contraction)
                Tytu.resize(r2, Ty1.n(), Ty1.r2());
                axpby_ortho_contract3(Mmt, unfold_right(Ty1), unfold_right(Tytu));
                //EigenMap(unfold_right(Tytu)) = ConstEigenMap(Mmt) * ConstEigenMap(unfold_right(Ty1));

                // calculate lower half of new Tyt: Tytl <- R *1 Ty1 (mode-1 contraction)
                Tytl.resize(R.r1(), Ty1.n(), Ty1.r2());
                axpby_ortho_contract3(R, unfold_right(Ty1), unfold_right(Tytl));
                //EigenMap(unfold_right(Tytl)) = ConstEigenMap(R) * ConstEigenMap(unfold_right(Ty1));

                // save this iteration's result into TTy
                Tdummy[1].resize(Tdummy[0].r2(), Ty1.n(), Ty1.r2()); // update dimension_dummy
                Tdummy = TTy.setSubTensors(k, std::move(Tdummy));

            } // end loop

            // calculate TTy[d-1] <- (x_last_core + Tytu ; Tytl)
            internal::t3_axpy((T)1, x_last_core, Tytu);
            internal::t3_concat1(Tytu, Tytl, x_last_core);
            TTy.setSubTensor(d-1, std::move(x_last_core));
        }

        
        // implement TT axpby_rightOrtho
        template <typename T>
        void axpby_rightOrthogonalize(T alpha, const TensorTrain<T>& TTx_ortho, T beta, TensorTrain<T>& TTy)
        {
            const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();
            
            const auto& TTx = TTx_ortho;
            const int d = TTx.dimensions().size(); // order d
            std::vector<Tensor3<T>> Tdummy(2); // dummy 3Tensor to update TTy with: {Tdummy[0], Tdummy[1]} = {dimension_dummy, Tz}

            // scale first tensor cores
            Tensor3<T> x_first_core;
            copy(TTx.subTensor(0), x_first_core);
            internal::t3_scale(alpha, x_first_core);
            copy(TTy.subTensor(0), Tdummy[0]);
            internal::t3_scale(beta, Tdummy[0]);
            if (d == 1)
            {
                internal::t3_axpy((T)1, x_first_core, Tdummy[0]);
                TTy.setSubTensor(0, std::move(Tdummy[0]));
                return;
            }
            Tdummy[0] = TTy.setSubTensor(0, std::move(Tdummy[0]));

            Tensor3<T> Tytl;        // left half of Y tilde
            Tensor3<T> Tytr;        // right half of Y tilde
            Tensor2<T> Mmt;     // Mmt = Mytl * Mx^adj (= Myt * Mxt^adj), where adj is the adjoint matrix (conjugate transpose)
            Tensor3<T> Ttmp;        // short-lived 3Tensor: to take QR decomposition of
            
            // initialize Tytl, Tytl for k == d-1
            copy(TTy.subTensor(d-1), Tytl);
            Tytr.resize(TTy.subTensor(d-1).r1(), TTy.subTensor(d-1).n(), 0);

            for (int k = d - 1; k > 0; k--)
            {
                // convenience references
                const Tensor3<T>& Tx =  TTx.subTensor(k);
                const Tensor3<T>& Ty1 = TTy.subTensor(k-1);
                // convenience variables for unfolded tensors
                ConstTensor2View<T> Mx = unfold_right(Tx);
                Tensor2View<T> Mytl = unfold_right(Tytl);

                // Mmt <- Mytl * Mx^adj
                axpby_ortho_contract1r(Mytl, Mx, Mmt);
                
                // Mytl <- Mytl - Mmt * Mx
                axpby_ortho_contract2(Mmt, Mx, Mytl);

                // Ttmp <- concat(Tytl, Tytr, dim=3)
                internal::t3_concat3(Tytl, Tytr, Ttmp);

                // [L, Q] <- QR(Mtmp)
                const int r2 = Tytr.r2() + Tytl.r2(); // r_k + st_k
                const int n_k = Tx.n(); // n_k
                const int r1 = Tx.r1(); // r_{k-1}
                assert(r2*n_k - r1 >= 0);
                const auto& [L, Q] = internal::normalize_qb(unfold_right(Ttmp), false, T(0), r2*n_k - r1, true);
                
                // calculate Tz: Tdummy[1] <- ((Tx, 0); Q)
                internal::concat1_X0Q(Tx, Q, Tdummy[1]);

                // calculate left half of new Tyt: Tytl <- Ty1 *3 Mmt (mode-3 contraction)
                Tytl.resize(Ty1.r1(), Ty1.n(), r1);
                axpby_ortho_contract3(unfold_left(Ty1), Mmt, unfold_left(Tytl));
                //EigenMap(unfold_left(Tytl)) = ConstEigenMap(unfold_left(Ty1)) * ConstEigenMap(Mmt);

                // calculate right half of new Tyt: Tytr <- Ty1 *3 L (mode-3 contraction)
                Tytr.resize(Ty1.r1(), Ty1.n(), L.r2());
                axpby_ortho_contract3(unfold_left(Ty1), L, unfold_left(Tytr));
                //EigenMap(unfold_left(Tytr)) = ConstEigenMap(unfold_left(Ty1)) * ConstEigenMap(L);

                // save result into TTy
                Tdummy[0].resize(TTy.subTensor(k-1).r1(), TTy.subTensor(k-1).n(), Tdummy[1].r1()); // update dimension_dummy
                Tdummy = TTy.setSubTensors(k-1, std::move(Tdummy));

            } // end loop

            // calculate TTy[0] <- (x_first_core + Tytl, Tytr)
            internal::t3_axpy(T(1), x_first_core, Tytl);
            internal::t3_concat3(Tytl, Tytr, x_first_core);
            TTy.setSubTensor(0, std::move(x_first_core));
        }


        // implement TT axpby_normalized
        template <typename T>
        T axpby_normalized(T alpha, const TensorTrain<T>& TTx_ortho, T beta, TensorTrain<T>& TTy, T rankTolerance, int maxRank)
        {
            const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();

            const TT_Orthogonality x_ortho = TTx_ortho.isOrthogonal();
            
            // check that x_ortho != none and TTx is actually orthogonalized
#ifndef NDEBUG
            if( !internal::is_normalized(TTx_ortho, x_ortho) )
              throw std::invalid_argument("TensorTrain TTx not orthogonalized on input to axpby_normalized!");
#endif
            
            T gamma;
            if (x_ortho == TT_Orthogonality::left)
            {
                internal::axpby_leftOrthogonalize(alpha, TTx_ortho, beta, TTy); // orthogonalization sweep left to right
                gamma = rightNormalize(TTy, rankTolerance, maxRank);            // compression sweep right to left
            }
            else //if (x_ortho == TT_Orthogonality::right)
            {
                internal::axpby_rightOrthogonalize(alpha, TTx_ortho, beta, TTy); // orthogonalization sweep right to left
                gamma = leftNormalize(TTy, rankTolerance, maxRank);              // compression sweep left to right
            }
            return gamma;
        }

    
    } // namespace internal

} // namespace PITTS

#endif // PITTS_TENSORTRAIN_AXPBY_NORMALIZED_IMPL_HPP
