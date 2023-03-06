/*! @file pitts_tensortrain_axpby_normalized_impl.hpp
* @brief addition for simple tensor train format where one of the tensors is normalized
* @author Manuel Joey Becklas <Manuel.Becklas@DLR.de>
* @date 2022-09-06
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
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
#include "pitts_tensor3_fold.hpp"
#include "pitts_tensor3_unfold.hpp"
#include "pitts_timer.hpp"

namespace PITTS
{
    namespace internal
    {
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
            const int nChunk = x.nChunks();
            const int r2 = x.r2();

            const auto timer = PITTS::performance::createScopedTimer<Tensor3<T>>(
                {{"r1", "n", "r2"}, {r1, n, r2}},                                     // arguments
                {{r1*n*r2*kernel_info::FMA<T>()},                                     // flops
                 {r1*n*r2*kernel_info::Load<T>() + r1*n*r2*kernel_info::Update<T>()}} // data: load x ; update y
            );
            
#pragma omp parallel for collapse(3) schedule(static) if(r1*n*r2 > 500)
            for(int i2 = 0; i2 < r2; i2++)
                for(int jChunk = 0; jChunk < nChunk; jChunk++)
                    for(int i1 = 0; i1 < r1; i1++)
                        fmadd(a, x.chunk(i1, jChunk, i2), y.chunk(i1, jChunk, i2));               
        }


        /**
         * @brief Compute z <- x^tr * y.
         * 
         * @tparam T 
         * @param x [in] Tensor2
         * @param y [in] Tensor2
         * @param z [out] Tensor2
         */
        template <typename T>
        inline void xtryz(const Tensor2<T>& x, const Tensor2<T>& y, Tensor2<T>& z)
        {
            const int r1  = x.r1();
            const int xr2 = x.r2();
            const int yr2 = y.r2();

            assert(r1 == y.r1());

            const auto timer = PITTS::performance::createScopedTimer<Tensor3<T>>(
                {{"r1", "x.r2", "y.r2"}, {r1, xr2, yr2}},                                      // arguments
                {{r1*xr2*yr2*kernel_info::FMA<T>()},                                           // flops
                 {(r1*xr2 + r1*yr2)*kernel_info::Load<T>() + xr2*yr2*kernel_info::Store<T>()}} // data: load x,y ; store z
            );

            z.resize(xr2, yr2);
            EigenMap(z) = ConstEigenMap(x).adjoint() * ConstEigenMap(y);
        }


        /**
         * @brief Compute z <- x * y^tr. 
         * 
         * @tparam T 
         * @param x [in] Tensor2
         * @param y [in] Tensor2
         * @param z [out] Tensor2
         */
        template <typename T>
        inline void xytrz(const Tensor2<T>& x, const Tensor2<T>& y, Tensor2<T>& z)
        {
            const int xr1  = x.r1();
            const int yr1  = y.r1();
            const int r2 = x.r2();

            assert(r2 == y.r2());

            const auto timer = PITTS::performance::createScopedTimer<Tensor3<T>>(
                {{"x.r1", "y.r1", "r2"}, {xr1, yr1, r2}},                                      // arguments
                {{xr1*yr1*r2*kernel_info::FMA<T>()},                                           // flops
                 {(xr1*r2 + yr1*r2)*kernel_info::Load<T>() + xr1*yr1*kernel_info::Store<T>()}} // data: load x,y ; store z
            );

            z.resize(xr1, yr1);
            EigenMap(z) = ConstEigenMap(x) * ConstEigenMap(y).adjoint();
        }


        /**
         * @brief Compute C <- C - A * B.
         * "fnmadd of Tensor2's"
         * 
         * @tparam T 
         * @param A [in] Tensor2
         * @param B [in] Tensor2
         * @param C [in/out] Tensor2
         */
        template<typename T>
        inline void t2_fnmadd(const Tensor2<T>& A, const Tensor2<T>& B, Tensor2<T>& C)
        {
            const int r1 = C.r1();
            const int r2 = C.r2();
            const int c  = A.r2();

            assert(r1 == A.r1());
            assert(r2 == B.r2());
            assert(c == B.r1());

            const auto timer = PITTS::performance::createScopedTimer<Tensor3<T>>(
                {{"r1", "c", "r2"}, {r1, c, r2}},                                                // arguments
                {{r1*r2*c*kernel_info::FMA<T>()},                                                // flops
                 {(r1*c + c*r2 + r1*r2)*kernel_info::Load<T>() + r1*r2*kernel_info::Store<T>()}} // data: load A,B,C ; store D
            );

            EigenMap(C) -= ConstEigenMap(A) * ConstEigenMap(B);
        }


        /**
         * @brief Special falvor of C <- concat(Le, Ri, dim=3), the concatination of Le and Ri along the third dimension.
         * 
         * 1. Le is interpreted as a right-unfolded 3Tensor
         * 2. Le and Ri are concatinated as if they were 3Tensors
         * 3. The result is written into C as the right-unfolded result
         * 
         * @tparam T 
         * @param Le    [in]  2Tensor (right-unfolded)
         * @param Ri    [in]  3Tensor
         * @param C     [out] 2Tensor (right-unfolded)
         */
        template <typename T>
        inline void t2t3_concat3(const Tensor2<T>& Le, const Tensor3<T>& Ri, Tensor2<T>& C)
        {
            const int r1     = Ri.r1();
            const int n      = Ri.n();
            const int r2l    = Le.r2() / n;
            const int r2r    = Ri.r2();

            assert(Le.r2() % n == 0);
            assert(Ri.r1() == Le.r1());

            C.resize(r1, n*(r2l + r2r));

            const auto timer = PITTS::performance::createScopedTimer<Tensor3<T>>(
                {{"r1", "n", "r2left", "r2right"}, {r1, n, r2l, r2r}},                                    // arguments
                {{(r1*n*r2l + r1*n*r2r)*kernel_info::NoOp<T>()},                                          // flops
                 {(r1*n*r2l + r1*n*r2r)*kernel_info::Load<T>() + r1*n*(r2l+r2r)*kernel_info::Store<T>()}} // data: load Le,Ri ; store C
            );

#pragma omp parallel
{
#pragma omp for schedule(static) collapse(2) nowait
            for (int i2 = 0; i2 < r2l*n; i2++)
                for (int i1 = 0; i1 < r1; i1++)
                {
                    C(i1, i2) = Le(i1, i2);
                }
#pragma omp for schedule(static) collapse(2) nowait // or 3 ?
            for (int i2 = 0; i2 < r2r; i2++)
                for(int j = 0; j < n; j++)
                {
                    for (int i1 = 0; i1 < r1; i1++)
                    {
                        C(i1, j + (i2+r2l)*n) = Ri(i1, j, i2);
                    }
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
            const int nChunk = Le.nChunks();
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
                for(int jChunk = 0; jChunk < nChunk; jChunk++)
                {
                    for (int i1 = 0; i1 < r1; i1++)
                    {
                        C.chunk(i1, jChunk, i2) = Le.chunk(i1, jChunk, i2);
                    }
                }
#pragma omp for schedule(static) collapse(2) nowait
            for (int i2 = 0; i2 < r2r; i2++)
                for(int jChunk = 0; jChunk < nChunk; jChunk++)
                {
                    for (int i1 = 0; i1 < r1; i1++)
                    {
                        C.chunk(i1, jChunk, i2 + r2l) = Ri.chunk(i1, jChunk, i2);
                    }
                }
}
        }


        /**
         * @brief Compute C <- concat(Le, 0, dim=3)
         * the concatination of Le and a 0-Tensor3 along the third dimensions, 0 of dimension Le.r1 x Le.n x r2Ri
         * 
         * @tparam T 
         * @param Le    [in] Tensor3
         * @param r1Ri  r2-dimension of 0-Tensor3
         * @param C     [out] Tensor3
         */
        template <typename T>
        inline void t3_concat3_w0(const Tensor3<T>& Le, const int r2Ri, Tensor3<T>& C)
        {
            const int r1     = Le.r1();
            const int n      = Le.n();
            const int nChunk = Le.nChunks();
            const int r2l    = Le.r2();
            const int r2r    = r2Ri;

            const auto timer = PITTS::performance::createScopedTimer<Tensor3<T>>(
                {{"r1", "n", "r2left", "r2right"}, {r1, n, r2l, r2r}},                       // arguments
                {{r1*n*r2l*kernel_info::NoOp<T>()},                                          // flops
                 {r1*n*r2l*kernel_info::Load<T>() + r1*n*(r2l+r2r)*kernel_info::Store<T>()}} // data: load Le ; store C
            );

            C.resize(r1, n, r2l + r2r);

#pragma omp parallel
{
#pragma omp for schedule(static) collapse(2) nowait
            for (int i2 = 0; i2 < r2l; i2++)
                for (int jChunk = 0; jChunk < nChunk; jChunk++)
                {
                    for (int i1 = 0; i1 < r1; i1++)
                    {
                        C.chunk(i1, jChunk, i2) = Le.chunk(i1, jChunk, i2);
                    }
                }
#pragma omp for schedule(static) collapse(2) nowait
            for (int i2 = 0; i2 < r2r; i2++)
                for (int jChunk = 0; jChunk < nChunk; jChunk++)
                {
                    for (int i1 = 0; i1 < r1; i1++)
                    {
                        C.chunk(i1, jChunk, i2 + r2l) = Chunk<T>{};
                    }
                }
}
        }


        /**
         * @brief Special flavor of C <- concat(Up, Lo, dim=1), the concationation of Up and Lo along the first dimension.
         * 
         * 1. Up is interpreted as a left-unfolded 3Tensor of fitting dimension
         * 2. Up and Lo are concatted as if they were 3Tensors
         * 3. The result is written into C as the left-unfolded result
         * 
         * @tparam T 
         * @param Up    [in]  2Tensor (left-unfolded)
         * @param Lo    [in]  3Tensor
         * @param C     [out] 2Tensor (left-unfolded)
         */
        template <typename T>
        inline void t2t3_concat1(const Tensor2<T>& Up, const Tensor3<T>& Lo, Tensor2<T>& C)
        {
            const int n     = Lo.n();
            const int r1u   = Up.r1() / n;
            const int r1l   = Lo.r1();
            const int r2    = Lo.r2();

            assert(Up.r1() % n == 0);
            assert(Lo.r2() == Up.r2());

            const auto timer = PITTS::performance::createScopedTimer<Tensor3<T>>(
                {{"r1upp", "r1low", "n", "r2"}, {r1u, r1l, n, r2}},                                       // arguments
                {{(r1u*n*r2 + r1l*n*r2)*kernel_info::NoOp<T>()},                                          // flops
                 {(r1u*n*r2 + r1l*n*r2)*kernel_info::Load<T>() + (r1u+r1l)*n*r2*kernel_info::Store<T>()}} // data: load Up,Lo ; store C
            );

            C.resize((r1u + r1l)*n, r2);

#pragma omp parallel for schedule(static) collapse(2)
            for (int i2 = 0; i2 < r2; i2++)
            {
                for (int j = 0; j < n; j++)             // might be beneficial to unroll (to at least half chunk size)
                {                                       // but maybe that's too much pressure on registers...
                    for (int i1 = 0; i1 < r1u; i1++)    // or/and unmerge loops (do first col of up, then add col of Lo in gaps)
                    {
                        C(i1 + j*(r1u+r1l), i2) = Up(i1 + j*r1u, i2);
                    }
                    for (int i1 = 0; i1 < r1l; i1++)
                    {
                        C(i1 + j*(r1u+r1l) + r1u, i2) = Lo(i1, j, i2);
                    }
                }
            }
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
            const int nChunk = Up.nChunks();
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
                for (int jChunk = 0; jChunk < nChunk; jChunk++)
                {
                    for (int i1 = 0; i1 < r1u; i1++)
                    {
                        C.chunk(i1, jChunk, i2) = Up.chunk(i1, jChunk, i2);
                    }
                    for (int i1 = 0; i1 < r1l; i1++)
                    {
                        C.chunk(i1 + r1u, jChunk, i2) = Lo.chunk(i1, jChunk, i2);
                    }
                }
        }


        /**
         * @brief Compute C <- concat(Up, 0, dim=1), 
         * the concationation of Up and a 0-Tensor3 along the first dimension, 0 of dimension r1Lo x Up.n x Up.r2
         * 
         * @tparam T 
         * @param Up    [in] Tensor3
         * @param r1Lo  r1-dimension of 0-Tensor3
         * @param C     [out] Tensor3
         */
        template <typename T>
        inline void t3_concat1_w0(const Tensor3<T>& Up, const int r1Lo, Tensor3<T>& C)
        {
            const int r1u    = Up.r1();
            const int r1l    = r1Lo;
            const int n      = Up.n();
            const int nChunk = Up.nChunks();
            const int r2     = Up.r2();

            const auto timer = PITTS::performance::createScopedTimer<Tensor3<T>>(
                {{"r1upp", "r1low", "n", "r2"}, {r1u, r1l, n, r2}},                          // arguments
                {{r1u*n*r2*kernel_info::NoOp<T>()},                                          // flops
                 {r1u*n*r2*kernel_info::Load<T>() + (r1u+r1l)*n*r2*kernel_info::Store<T>()}} // data: load Up ; store C
            );

            C.resize(r1u + r1l, n, r2);
            
#pragma omp parallel for schedule(static) collapse(2)
            for (int i2 = 0; i2 < r2; i2++)
                for (int jChunk = 0; jChunk < nChunk; jChunk++)
                {
                    for (int i1 = 0; i1 < r1u; i1++)
                    {
                        C.chunk(i1, jChunk, i2) = Up.chunk(i1, jChunk, i2);
                    }
                    for (int i1 = 0; i1 < r1l; i1++)
                    {
                        C.chunk(i1 + r1u, jChunk, i2) = Chunk<T>{};
                    }
                }
        }


        // implement TT is_normalized
        template<typename T>
        bool is_normalized(const TensorTrain<T>& A, TT_Orthogonality orthog, double eps)
        {
            if (orthog == TT_Orthogonality::none) return false;

            Tensor2<T> core;
            for (int i = 0; i < A.dimensions().size() - 1; i++)
            {
                int i_ = (orthog == TT_Orthogonality::left) ? i : i + 1; // shift range by one for right-orthogonality

                if (orthog == TT_Orthogonality::left)
                    unfold_left(A.subTensor(i_), core);
                else
                    unfold_right(A.subTensor(i_), core);

                using EigenMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
                auto mat = ConstEigenMap(core);
                EigenMatrix orth;
                if (orthog == TT_Orthogonality::left)
                    orth = mat.transpose() * mat;
                else
                    orth = mat * mat.transpose();
                EigenMatrix orthErr = orth - EigenMatrix::Identity(orth.cols(), orth.rows());
                if (orthErr.array().abs().maxCoeff() > eps)
                {
                  std::cout << "Error: Sub-Tensor " << i_ << " should be " << (orthog == TT_Orthogonality::left ? "left" : "right") << "-orthogonal I-V^TV is:\n";
                  std::cout << orthErr << "\n";
                  std::cout << "And the tolerance is: " << eps << "\n";
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
            std::vector<Tensor3<T>> Tdummy(2); // dummy 3Tensor to update TTy with: [Tdummy[0], Tdummy[1] = [Tz, dimension_dummy]

            // scale last tensor cores
            Tensor3<T> x_last_core;
            copy(TTx.subTensor(d-1), x_last_core);
            internal::t3_scale(alpha, x_last_core);
            //internal::t3_scale(beta, y_cores[d-1]);
            copy(TTy.subTensor(d-1), Tdummy[0]);
            internal::t3_scale(beta, Tdummy[0]);
            if (d == 1) // special case
            {
                internal::t3_axpy((T)1, x_last_core, Tdummy[0]);
                TTy.setSubTensor(d-1, std::move(Tdummy[0]));
                return;
            }
            Tdummy[0] = TTy.setSubTensor(d-1, std::move(Tdummy[0]));

            //
            // Note: In the loop sweep, a few mem buffers could be reused -> giving better memory usage
            //
            Tensor3<T> Tytu;    // upper half of Y tilde
            Tensor3<T> Tytl;    // lower half of Y tilde
            Tensor2<T> Mmt;     // Mx^tr * Mytu (= Mxt^tr * Myt)

            Tensor2<T> Mx;      // Tx left-unfolded
            Tensor2<T> Mytu;    // Tytu left-unfolded
            //Tensor2<T> Mytl;    // Tytl left-unfolded

            Tensor2<T> Mtmp;    // short-lived 2Tensor: holding result Myt - Mxt * Mmt to take QR decomposition of
            //Tensor2<T> Mtmpu;   // short-lived 2Tensor: calculating upper half of Mtmp into here
            Tensor3<T> Txt;     // short-lived 3Tensor: X tilde (= concat1(Tx, 0))
            Tensor3<T> TQ;      // short-lived 3Tensor: Q left-folded
            
            // initialize Tyt(u/l) for k == 0
            copy(TTy.subTensor(0), Tytu);
            Tytl.resize(0, TTy.subTensor(0).n(), TTy.subTensor(0).r2());

            for (int k = 0; k < d - 1; k++)
            {
                // convenience references
                const Tensor3<T>& Ty1 = TTy.subTensor(k+1);
                const Tensor3<T>& Tx =  TTx.subTensor(k);

                //
                // Note: If Mxt is square matrix, most of the calculation (especially QR) can be skipped
                //

                // copy Mx <- Tx(:,: x :), Mytu <- Tytu(:,: x :)
                unfold_left(Tx, Mx);
                unfold_left(Tytu, Mytu);
                //unfold_left(Tytl, Mytl);

                // Mmt <- Mx^tr * Mytu
                internal::xtryz(Mx, Mytu, Mmt);
                
                // Mtmpu <- Mytu - Mx * Mmt
                internal::t2_fnmadd(Mx, Mmt, Mytu); // Mtmpu can be Mytu (change t2_fnmadd implementatio) ------------ 

                // concatinate Mtmp <- concat(Mtmpu, Tytl, dim=1)
                internal::t2t3_concat1(Mytu, Tytl, Mtmp);

                // [Q, R] <- QR(Mtmp)
                const int r1 = Tytu.r1() + Tytl.r1(); // r_{k-1} + st_{k-1}
                const int n_k = Tx.n(); // n_k
                const int r2 = Tx.r2(); // r_k
                assert(r1*n_k - r2 >= 0);
                const auto& [Q, R] = internal::normalize_qb(Mtmp, true, T(0), r1*n_k - r2, true);
                
                // TQ <- fold(Q), Txt <- concat(Tx, 0, dim=1), Tdummy[0] <- concat(Txt, TQ, dim=3)
                fold_left(Q, n_k, TQ);
                internal::t3_concat1_w0(Tx, Tytl.r1(), Txt);
                internal::t3_concat3(Txt, TQ, Tdummy[0]); // if resizing without destroying data possible -> could just append TQ to Txt and then swap Txt and Tz

                // calculate upper half of new Tyt: Tytu <- Mmt *1 Ty1 (mode-1 contraction)
                internal::normalize_contract1(Mmt, Ty1, Tytu);

                // calculate lower half of new Tyt: Tytl <- R *1 Ty1 (mode-1 contraction)
                internal::normalize_contract1(R, Ty1, Tytl);

                // save this iteration's result into TTy
                Tdummy[1].resize(Tdummy[0].r2(), TTy.subTensor(k+1).n(), TTy.subTensor(k+1).r2()); // update dimension_dummy
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
            std::vector<Tensor3<T>> Tdummy(2); // dummy 3Tensor to update TTy with: [Tdummy[0], Tdummy[1] = [dimension_dummy, Tz]

            // scale first tensor cores
            Tensor3<T> x_first_core;
            copy(TTx.subTensor(0), x_first_core);
            internal::t3_scale(alpha, x_first_core);
            //internal::t3_scale(beta, y_cores[0]);
            copy(TTy.subTensor(0), Tdummy[0]);
            internal::t3_scale(beta, Tdummy[0]);
            if (d == 1) // special case
            {
                internal::t3_axpy((T)1, x_first_core, Tdummy[0]);
                TTy.setSubTensor(0, std::move(Tdummy[0]));
                return;
            }
            Tdummy[0] = TTy.setSubTensor(0, std::move(Tdummy[0]));

            Tensor3<T> Tytl;    // left half of Y tilde
            Tensor3<T> Tytr;    // right half of Y tilde
            Tensor2<T> Mmt;     // Mx^tr * Mytu (= Mxt^tr * Myt)

            Tensor2<T> Mx;      // Tx right-unfolded
            Tensor2<T> Mytl;    // Tytl right-unfolded
            Tensor2<T> Mytr;    // Tytr right-unfolded

            Tensor2<T> Mtmp;    // short-lived 2Tensor: holding result Myt - Mxt * Mmt to take QR decomposition of
            //Tensor2<T> Mtmpl;   // short-lived 2Tensor: calculating left half of Mtmp into here
            Tensor3<T> Txt;     // short-lived 3Tensor: X tilde (= concat1(Tx, 0))
            Tensor3<T> TQ;      // short-lived 3Tensor: Q right-folded
            
            // initialize Tyt(l/r) for k == d-1
            copy(TTy.subTensor(d-1), Tytl);
            Tytr.resize(TTy.subTensor(d-1).r1(), TTy.subTensor(d-1).n(), 0);

            for (int k = d - 1; k > 0; k--)
            {
                // convenience references
                const Tensor3<T>& Tx =  TTx.subTensor(k);
                const Tensor3<T>& Ty1 = TTy.subTensor(k-1);

                // Mx <- Tx(: x :,:), Mytl <- Tytl(: x :,:), Mytr <- Tytr(: x :,:)
                unfold_right(Tx, Mx);
                unfold_right(Tytl, Mytl);
                unfold_right(Tytr, Mytr);

                // Mmt <- Mytl * Mx^tr
                internal::xytrz(Mytl, Mx, Mmt);
                
                // Mtmpl <- Mytl - Mmt * Mx
                internal::t2_fnmadd(Mmt, Mx, Mytl);

                // concatinate Mtmp <- concat(Mtmpl, Tytr, dim=3)
                internal::t2t3_concat3(Mytl, Tytr, Mtmp);

                // [L, Q] <- QR(Mtmp)
                const int r2 = Tytr.r2() + Tytl.r2(); // r_k + st_k
                const int n_k = Tx.n(); // n_k
                const int r1 = Tx.r1(); // r_{k-1}
                assert(r2*n_k - r1 >= 0);
                const auto& [L, Q] = internal::normalize_qb(Mtmp, false, T(0), r2*n_k - r1, true);
                
                // TQ <- fold(Q), Txt - concat(Tx, 0, dim=3), Tdummy[1] <- concat(Txt, TQ, dim=1)
                fold_right(Q, n_k, TQ);
                internal::t3_concat3_w0(Tx, Tytr.r2(), Txt);
                internal::t3_concat1(Txt, TQ, Tdummy[1]);

                // calculate left half of new Tyt: Tytl <- Ty1 *3 Mmt (mode-3 contraction)
                internal::normalize_contract2(Ty1, Mmt, Tytl);

                // calculate right half of new Tyt: Tytr <- Tyt *3 L (mode-3 contraction)
                internal::normalize_contract2(Ty1, L, Tytr);

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