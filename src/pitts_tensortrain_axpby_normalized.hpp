/*! @file pitts_tensortrain_axpby_normalized.hpp
* @brief addition for simple tensor train format where one of the tensors is normalized
* @author Manuel Joey Becklas <Manuel.Becklas@DLR.de>
* @date 2022-09-06
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_AXPBY_NORMALIZED_HPP
#define PITTS_TENSORTRAIN_AXPBY_NORMALIZED_HPP

//#define VERBOSE

// includes
#include "pitts_tensortrain.hpp"
#include "pitts_tensortrain_norm.hpp"
#include "pitts_tensortrain_normalize.hpp"
#include "pitts_tensor2.hpp"
#include "pitts_tensor3_split.hpp"

#define FIXED_FLOAT(f, prec) std::fixed << std::setprecision(prec) << (f < 0 ? "" : " ") << f

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
        inline void zxytr(const Tensor2<T>& x, const Tensor2<T>& y, Tensor2<T>& z)
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
         * @brief Compute D <- C - A * B.
         * "fnmadd of Tensor2's"
         * 
         * @tparam T 
         * @param A [in] Tensor2
         * @param B [in] Tensor2
         * @param C [in] Tensor2
         * @param D [out] Tensor2
         */
        template<typename T>
        inline void t2_fnmadd(const Tensor2<T>& A, const Tensor2<T>& B, const Tensor2<T>& C, Tensor2<T>& D)
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

            D.resize(r1, r2);
            EigenMap(D) = ConstEigenMap(C) - ConstEigenMap(A) * ConstEigenMap(B);
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
         * 1. Lo is interpreted as a left-unfolded 3Tensor of fitting dimension
         * 2. Lo and Up are concatted as if they were 3Tensors
         * 3. The result is written into C as the left-unfolded result
         * 
         * @tparam T 
         * @param Up [in]   2Tensor (left-unfolded)
         * @param Lo [in]   3Tensor
         * @param C  [out]  2Tensor (left-unfolded)
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

            C.resize((r1u + r1l)*n, r2);

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


        /**
         * @brief Returns if the tensor train A is (left or right)-orthogonal (up to some tolerance).
         * 
         * @tparam T        underlying type
         * @param A         TensorTrain<T> object
         * @param orthog    what type of orthogonality to check foe
         * @return true     if A passes the orthogonality test
         * @return false    if A fails the orthogonality test
         */
        template<typename T>
        bool is_normalized(const TensorTrain<T>& A, TT_Orthogonality orthog, double eps = std::sqrt(std::numeric_limits<T>::epsilon()))
        {
            if (orthog == TT_Orthogonality::none) return false;

            Tensor2<T> core;
            for (int i = 0; i < A.dimensions().size() - 1; i++)
            {
                int i_ = (orthog == TT_Orthogonality::left) ? i : i + 1; // shift range by one for rigth orthogonality

                if (orthog == TT_Orthogonality::left)
                    unfold_left(A.subTensor(i_), core);
                else
                    unfold_right(A.subTensor(i_), core);

                using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
                Matrix mat = Eigen::Map<Matrix>(&core(0, 0), core.r1(), core.r2());
                Matrix orth;
                if (orthog == TT_Orthogonality::left)
                    orth = mat.transpose() * mat;
                else
                    orth = mat * mat.transpose();
                if ((orth - Matrix::Identity(orth.cols(), orth.rows())).norm() > eps)
                    return false;
            }
            return true;
        }
        

        /*
        * print the 2-tensor to command line
        */
        template <typename T>
        static void quickndirty_visualizeMAT(const Tensor2<T>& G, int prec = 4)
        {
            std::cout << "dimensions: " << G.r1() << " x "<< G.r2() << std::endl;
            if (G.r1() == 0 || G.r2() == 0) goto _return;
            for (int i1  = 0; i1 < G.r1(); i1++)
            {
                for (int i2 = 0; i2 < G.r2(); i2++)
                {
                    std::cout << FIXED_FLOAT(G(i1, i2), prec) << '\t';
                }
                std::cout << std::endl;
            }
        _return:
            std::cout << "\n";
        }

        /*
        * print the 3-tensor to command line
        * 
        * @param prec    digits after decimal point to be displayed
        * 
        * @param fold    if fold == 1, combine r1 and n dimensions
        *                if fold == 2, combine n and r2 dimensions
        *                if fold == 0, don't fold, show depth (n) to the side
        */
        template <typename T>
        static void quickndirty_visualizeCORE(const Tensor3<T>& G, int prec = 4, int fold = 0)
        {
            std::cout << "dimensions: " << G.r1() << " x " << G.n() << " x "<< G.r2() << std::endl;
            if (fold == 1) // combine r1 and n dimension
            {
                for (int j = 0; j < G.n(); j++)
                {
                    for (int i1  = 0; i1 < G.r1(); i1++)
                    {
                        for (int i2 = 0; i2 < G.r2(); i2++)
                        {
                            std::cout << FIXED_FLOAT(G(i1, j, i2), prec) << '\t';
                        }
                        std::cout << std::endl;
                    }
                }
            }
            if (fold == 2) // combine n and r2 dimension
            {
                for (int i1  = 0; i1 < G.r1(); i1++)
                {
                    for (int i2 = 0; i2 < G.r2(); i2++)
                    {
                        for (int j = 0; j < G.n(); j++)
                        {
                            std::cout << FIXED_FLOAT(G(i1, j, i2), prec) << '\t';
                        }
                    }
                    std::cout << std::endl;
                }
            }
            if (fold == 0)
            {
                for (int i1  = 0; i1 < G.r1(); i1++)
                {
                    for (int j = 0; j < G.n(); j++)
                    {
                        for (int i2 = 0; i2 < G.r2(); i2++)
                        {
                            std::cout << FIXED_FLOAT(G(i1, j, i2), prec) << '\t';
                        }
                        std::cout << '\t';
                    }
                    std::cout << std::endl;
                }
            }
            std::cout << "\n";
        }

        /*
        * print the complete tensor to command line
        * 
        * @param prec    digits after decimal point to be displayed
        * 
        * @param fold    if fold == 1, combine r1 and n dimensions
        *                if fold == 2, combine n and r2 dimensions
        *                if fold == 0, don't fold, show depth (n) to the side
        */
        template <typename T>
        static void quickndirty_visualizeTT(const TensorTrain<T>& TT, int prec = 4, int fold = 0)
        {
            for (int i = 0; i < TT.dimensions().size(); i++)
            {
                quickndirty_visualizeCORE(TT.subTensor(i), prec, fold);
            }
        }


        
        /**
         * @brief Left to right orthogonalization sweep for axpby_normalized function.
         * This performs the axbpy operation as well as the orthogonalization.
         * 
         * @tparam T            underlying data type
         * @param alpha         coefficient of tensor x, scalar value
         * @param TTx_ortho     tensor x in tensor train format, left-orthogonal
         * @param beta          coefficient of tensor y, scalar value
         * @param TTy           tensor y in tensor train format
         */
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
            Tensor2<T> Mytl;    // Tytl left-unfolded

            Tensor2<T> Mtmp;    // short-lived 2Tensor: holding result Myt - Mxt * Mmt to take QR decomposition of
            Tensor2<T> Mtmpu;   // short-lived 2Tensor: calculating upper half of Mtmp into here
            Tensor3<T> Txt;     // short-lived 3Tensor: X tilde (= concat1(Tx, 0))
            Tensor3<T> TQ;      // short-lived 3Tensor: Q left-unfolded
            
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

                // copy Mx <- Tx(:,: x :), Mytu <- Tytu(:,: x :), Mytl <- Tytl(:,: x :)
                unfold_left(Tx, Mx);
                unfold_left(Tytu, Mytu);
                //unfold_left(Tytl, Mytl);

                // Mmt <- Mx^tr * Mytu
                internal::xtryz(Mx, Mytu, Mmt);
                
                // Mtmpu <- Mytu - Mx * Mmt
                internal::t2_fnmadd(Mx, Mmt, Mytu, Mtmpu); // Mtmpu can be Mytu (change t2_fnmadd implementatio) ------------ 

                // concatinate Mtmp <- concat(Mtmpu, Tytl, dim=1)
                internal::t2t3_concat1(Mtmpu, Tytl, Mtmp);

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

            // calculate TTy[d-1] <- x_last_core + Tyt (componentwise)
            internal::t3_axpy((T)1, x_last_core, Tytu);
            internal::t3_concat1(Tytu, Tytl, x_last_core);
            TTy.setSubTensor(d-1, std::move(x_last_core));
        }

        
        /**
         * @brief Right to left orthogonalization sweep for axpby_normalized function.
         * This performs the axbpy operation as well as the orthogonalization.
         * 
         * @tparam T            underlying data type
         * @param alpha         coefficient of tensor x, scalar value
         * @param TTx_ortho     tensor x in tensor train format, right-orthogonal
         * @param beta          coefficient of tensor y, scalar value
         * @param TTy           tensor y in tensor train format
         */
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

            Tensor3<T> Tyt;             // temporary 3Tensor y (Y tilde)
            Tensor3<T> Txt;             // temporary 3Tensor x (X tilde)
            Tensor2<T> Mmt;             // temporary 2Tensor to hold result Myt * Mxt^tr

            Tensor2<T> Mtmp;            // temporary 2Tensor to hold result Myt - Mmt * Mxt
            Tensor3<T> Ttmpl;           // temporary 3Tensor to calculate left part (in r2-dim) of Tyt into
            Tensor3<T> Ttmpr;           // temporary 3Tensor to calculate right part (in r2-dim) of Tyt into

            Tensor2<T> Myt;             // 2Tensor copy of Tyt 
            Tensor2<T> Mxt;             // 2Tensor copy of Txt
            Tensor3<T> TQ;              // 3Tensor copy of Q
            
            // initialize Txt and Tyt for k == d-1
            copy(TTx.subTensor(d-1), Txt);
            copy(TTy.subTensor(d-1), Tyt);

            for (int k = d - 1; k > 0; k--)
            {
                // convenience references
                const Tensor3<T>& Ty1 = TTy.subTensor(k-1);
                const Tensor3<T>& Tx1 = (k == 1) ? x_first_core : TTx.subTensor(k-1);

                // Mxt <- Txt(: x :,:), Myt <- Tyt(: x :,:)
                unfold_right(Txt, Mxt);
                unfold_right(Tyt, Myt);

                // Mmt <- Myt * Mxt^tr
                internal::zxytr(Myt, Mxt, Mmt);
                
                // Mtmp <- Myt - Mmt * Mxt
                internal::t2_fnmadd(Mmt, Mxt, Myt, Mtmp);

                // [Q, B] <- QR(Mtmp)
                const int r2 = Txt.r2(); // common r2-dimension of Txt and Tyt
                const int n_k = Txt.n(); // n_k (n of Txt and Tyt)
                const int r1 = Txt.r1(); // r_{k-1} (r1 of Txt)
                assert(r2*n_k - r1 >= 0);
                const auto& [L, Q] = internal::normalize_qb(Mtmp, false, T(0), r2*n_k - r1, true);
                
                // TQ <- fold(Q)
                fold_right(Q, n_k, TQ);

                // Tdummy[1] <- concat(Txt, TQ, dim=1)
                internal::t3_concat1(Txt, TQ, Tdummy[1]);

                // calculate new Txt: Txt <- concat(Tx1, 0, dim=3), 0 of dimension Tx1.r1 x Tx1.n x L.r2
                internal::t3_concat3_w0(Tx1, Q.r1(), Txt);

                // calculate left half of new Tyt: Ttmpl <- Ty1 *3 Mmt (mode-3 contraction)
                internal::normalize_contract2(Ty1, Mmt, Ttmpl);

                // calculate right half of new Tyt: Ttmpr <- Tyt *3 L (mode-3 contraction)
                internal::normalize_contract2(Ty1, L, Ttmpr);

                // concatinate Tyt <- concat(Ttmpl, Ttempr, dim=3)
                internal::t3_concat3(Ttmpl, Ttmpr, Tyt);

                // save result into TTy
                Tdummy[0].resize(TTy.subTensor(k-1).r1(), TTy.subTensor(k-1).n(), Tdummy[1].r1()); // update dimension_dummy
                Tdummy = TTy.setSubTensors(k-1, std::move(Tdummy));

            } // end loop

            // calculate TTy[0] <- Txt + Tyt (componentwise)
            internal::t3_axpy(T(1), Tyt, Txt);
            TTy.setSubTensor(0, std::move(Txt));
        }


        /**
         * @brief Add scaled tensor trains, where one of them (x) is normalized.
         * 
         * Calculate gamma * y <- alpha * x + beta * y, 
         * such that the result y is orthogonalized and has frobenius norm 1.0
         * 
         * @warning Tensor x (TTx) must already be left- or right- orthogonal.
         * @warning This function doesn't check that tensor dimensions match nor special cases. Call the function axpby for that.
         * 
         * @tparam T underlying data type (double, complex, ...)
         * 
         * @param alpha         coefficient of tensor x, scalar value
         * @param TTx           orthogonalized tensor x in tensor train format
         * @param beta          coefficient of tensor y, scalar value
         * @param TTy           tensor y in tensor train format, result tensor
         * @param rankTolerance approxiamtion accuracy that is used to reduce the TTranks of the result
         * @param maxRank       maximal allowed TT-rank, enforced even if this violates the rankTolerance
         * @return              norm of the result tensor
         */
        template <typename T>
        T axpby_normalized(T alpha, const TensorTrain<T>& TTx_ortho, T beta, TensorTrain<T>& TTy, T rankTolerance = std::sqrt(std::numeric_limits<T>::epsilon()), int maxRank = 0x7fffffff)
        {
            const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();

            const TT_Orthogonality x_ortho = TTx_ortho.isOrthogonal();
            
            // check that x_ortho != none and TTx is actually orthogonalized
            assert(internal::is_normalized(TTx_ortho, x_ortho) == true);
            
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

#endif // PITTS_TENSORTRAIN_AXPBY_NORMALIZED_HPP
