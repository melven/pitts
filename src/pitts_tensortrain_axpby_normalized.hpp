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
         * @brief my version of wrapper for QB
         * 
         * @tparam T                underlying data type
         * @param M                 Tensor2 that is being decomposed
         * @param leftOrthog        true -> QR decomposition, false -> LQ decomposition
         * @param maxRank           maximal rank of Q (further cols are cut off)
         * @param rankTolerance     is IGNORED
         * @return                  [Q, R] resp. [L, Q]
         */
        template<typename T>
        auto m_normalize_qb(const Tensor2<T>& M, bool leftOrthog = true, int maxRank = std::numeric_limits<int>::max(), T rankTolerance = 0)
        {
            const auto timer = PITTS::timing::createScopedTimer<Tensor2<T>>();

            // get reasonable rank tolerance (ignoring passed value)        // was min(M.r1(), M.r2())
            const T rankTol = std::numeric_limits<decltype(rankTolerance)>::epsilon() * (M.r1() + M.r2()) / 2;
            rankTolerance = 16 * rankTol;

            // calculate QR decomposition
            using EigenMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
            const auto mapM = ConstEigenMap(M);
            auto qr = leftOrthog ?
                Eigen::ColPivHouseholderQR<EigenMatrix>(mapM) :
                Eigen::ColPivHouseholderQR<EigenMatrix>(mapM.transpose());

            //update rankTolerance (in case all pivot elements are tiny)
            rankTolerance = std::max(rankTolerance, rankTol / std::abs(qr.maxPivot()));
            // set threshold (which pivots to consider 0) for Eigen
            qr.setThreshold(rankTolerance);
            // rank of matrix
            const Eigen::Index rk = std::min(qr.rank(), Eigen::Index(maxRank));

            #ifdef VERBOSE
            std::cout << "threshold: \t\t" << FIXED_FLOAT(qr.threshold(), 20) << " is " << FIXED_FLOAT(qr.threshold()/std::numeric_limits<T>::epsilon(), 1) << " * machine_eps";
            (qr.threshold() == 16 * rankTol) ? std::cout << ", used RELATIVE rank tolerance\n" : std::cout << ", used ABSOLUTE rank tolerance\n";
            std::cout << "treshold * |maxpivot|: \t" << FIXED_FLOAT(qr.threshold() * std::abs(qr.maxPivot()), 20) << std::endl;
            std::cout << "biggest pivot element: \t" << FIXED_FLOAT(qr.maxPivot(), 20) << std::endl;
            std::cout << "matrix rank: " << qr.rank() << ", cut off at maxrank -> " << rk << std::endl;
            const EigenMatrix R_ = qr.matrixR();
            std::cout << "all pivot elements: ";
            for (int i = 0; i < std::min(R_.rows(), R_.cols()); i++)
                std::cout << FIXED_FLOAT(R_(i,i), 20) << '\t';
            std::cout << "\n\n";
            #endif

            // return result
            
            std::pair<Tensor2<T>,Tensor2<T>> result;
            result.first.resize(M.r1(), rk);
            result.second.resize(rk, M.r2());

            qr.householderQ().setLength(rk);
            const EigenMatrix R = qr.matrixR().topRows(rk).template triangularView<Eigen::Upper>();
            if( leftOrthog )
            {
                // return QR
                EigenMap(result.first) = qr.householderQ() * EigenMatrix::Identity(M.r1(), rk);
                EigenMap(result.second) = R * qr.colsPermutation().inverse();
            }
            else
            {
                // return LQ
                EigenMap(result.first) = (R * qr.colsPermutation().inverse()).transpose();
                EigenMap(result.second) = EigenMatrix::Identity(rk, M.r2()) * qr.householderQ().transpose();
            }

            return result;
        }


        /**
         * @brief Componentwise axpy for Tensor3 objects
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
         * @brief Returns if the tensor train A is (left or right)-orthogonal (up to some tolerance).
         * 
         * @tparam T        underlying type
         * @param A         TensorTrain<T> object
         * @param left      whether to check for left-orthogonality or right-orthogonality
         * @return true     if A passes the orthogonality test
         * @return false    if A fails the orthogonality test
         */
        template<typename T>
        bool is_normalized(const TensorTrain<T>& A /*,bool left = true*/)
        {
            // no timer because this function is only used in in an assert
            Tensor2<T> core;
            for (int i = 0; i < A.subTensors().size() - 1; i++)
            {
                unfold_left(A.subTensors()[i], core);

                using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
                Matrix mat = Eigen::Map<Matrix>(&core(0, 0), core.r1(), core.r2());
                Matrix orth;
                //if (left)
                    orth = mat.transpose() * mat;
                //else
                //    orth = mat * mat.transpose();
                if ((orth - Matrix::Identity(orth.cols(), orth.rows())).norm() > std::sqrt(std::numeric_limits<T>::epsilon()))
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
            for (const auto& core : TT.subTensors())
            {
                quickndirty_visualizeCORE(core, prec, fold);
            }
        }


        /**
         * @brief Compute z(:,:) <- x(:,: x :)^tr * y(:,: x :). 
         * "Z <- X TRanspose * Y"
         * 
         * @tparam T 
         * @param x [in] Tensor3
         * @param y [in] Tensor3
         * @param z [out] Tensor2
         */
        template <typename T>
        inline void zxtry(const Tensor3<T>& x, const Tensor3<T>& y, Tensor2<T>& z)
        {
            const int r1  = x.r1();
            const int n   = x.n();
            const int xr2 = x.r2();
            const int yr2 = y.r2();

            assert(r1 == y.r1());
            assert(n  == y.n());

            const auto timer = PITTS::performance::createScopedTimer<Tensor3<T>>(
                {{"r1", "n", "x.r2", "y.r2"}, {r1, n, xr2, yr2}},                                  // arguments
                {{r1*n*xr2*yr2*kernel_info::FMA<T>()},                                             // flops
                 {(r1*n*xr2 + r1*n*yr2)*kernel_info::Load<T>() + xr2*yr2*kernel_info::Store<T>()}} // data: load x,y ; store z
            );

            z.resize(xr2, yr2);

            for (int j = 0; j < yr2; j++)
            {
                for (int i = 0; i < xr2; i++)
                {
                    z(i, j) = 0;
                    for (int l = 0; l < n; l++)
                    {
                        for (int k = 0; k < r1; k++)
                        {
                            z(i,j) += x(k, l, i) * y(k, l, j);
                        }
                    }
                }
            }
        }


        /**
         * @brief Compute D <- C(:,: x :) - A(:,: x :) * B(: x :).
         * "fnmadd of Tensor2's, left-folding Tensor3's"
         * 
         * @tparam T 
         * @param A [in] Tensor3
         * @param B [in] Tensor3
         * @param C [in] Tensor2
         * @param D [out] Tensor2
         */
        template<typename T>
        inline void t3232_fnmadd(const Tensor3<T>& A, const Tensor2<T>& B, const Tensor3<T>& C, Tensor2<T>& D)
        {
            const int r1 = C.r1();
            const int n  = C.n();
            const int r2 = C.r2();
            const int c  = A.r2();

            assert(r1 == A.r1());
            assert(n == A.n());
            assert(r2 == B.r2());
            assert(c == B.r1());

            const auto timer = PITTS::performance::createScopedTimer<Tensor3<T>>(
                {{"r1", "n", "r2", "c"}, {r1, n, r2, c}},                                              // arguments
                {{r1*n*r2*c*kernel_info::FMA<T>()},                                                    // flops
                 {(r1*n*c + c*r2 + r1*n*r2)*kernel_info::Load<T>() + r1*n*r2*kernel_info::Store<T>()}} // data: load A,B,C ; store D
            );

            D.resize(r1 * n, r2);

            for (int j = 0; j < r2; j++)
            {
                for (int i2 = 0; i2 < n; i2++)
                {
                    for (int i1 = 0; i1 < r1; i1++)
                    {
                        D(i1 + i2*r1, j) = C(i1, i2, j);
                        for (int k = 0; k < c; k++)
                        {
                            D(i1 + i2*r1, j) -= A(i1, i2, k) * B(k, j);
                        }
                    }
                }
            }
        }

        
        /**
         * @brief Compute C <- concat(Le, Ri, dim=3), 
         * the concationation of Le and Ri along the third dimension - regarding Ri as a Tensor3 of fitting dimension (Le.r1 x Le.n x Ri.r2).
         * 
         * @tparam T 
         * @param Le [in] Tensor3
         * @param Ri [in] Tensor2
         * @param C [out] Tensor3
         */
        template <typename T>
        inline void t32_concat3(const Tensor3<T>& Le, const Tensor2<T>& Ri, Tensor3<T>& C)
        {
            const int r1  = Le.r1(); //c;
            const int n   = Le.n(); //n_k;
            const int r2l = Le.r2(); // r_k; actually left
            const int r2r = Ri.r2(); // st_k; actually right

            assert(r1 * n == Ri.r1());

            const auto timer = PITTS::performance::createScopedTimer<Tensor3<T>>(
                {{"r1", "n", "r2left", "r2right"}, {r1, n, r2l, r2r}},                                    // arguments
                {{(r1*n*r2l + r1*n*r2r)*kernel_info::NoOp<T>()},                                          // flops
                 {(r1*n*r2l + r1*n*r2r)*kernel_info::Load<T>() + r1*n*(r2l+r2r)*kernel_info::Store<T>()}} // data: load Le,Ri ; store C
            );

            C.resize(r1, n, r2l + r2r);

            for (int i2 = 0; i2 < r2l; i2++)
            {
                for (int j = 0; j < n; j++)
                {
                    for (int i1 = 0; i1 < r1; i1++)
                    {
                        C(i1, j, i2) = Le(i1, j, i2);
                    }
                }
            }
            for (int i2 = 0; i2 < r2r; i2++)
            {
                for (int j = 0; j < n; j++)
                {
                    for (int i1 = 0; i1 < r1; i1++)
                    {
                        C(i1, j, i2 + r2l) = Ri(i1 + j*r1, i2);
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
            const int r1u = Up.r1();
            const int r1l = Lo.r1();
            const int n   = Up.n();
            const int r2  = Up.r2();

            assert(n  == Lo.n());
            assert(r2 == Lo.r2());

            const auto timer = PITTS::performance::createScopedTimer<Tensor3<T>>(
                {{"r1upp", "r1low", "n", "r2"}, {r1u, r1l, n, r2}},                                       // arguments
                {{(r1u*n*r2 + r1l*n*r2)*kernel_info::NoOp<T>()},                                          // flops
                 {(r1u*n*r2 + r1l*n*r2)*kernel_info::Load<T>() + (r1u+r1l)*n*r2*kernel_info::Store<T>()}} // data: load Up,Lo ; store C
            );

            C.resize(r1u + r1l, n, r2);
            
            for (int i2 = 0; i2 < r2; i2++)
            {
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
            const int r1u = Up.r1();
            const int r1l = r1Lo;
            const int n   = Up.n();
            const int r2  = Up.r2();

            const auto timer = PITTS::performance::createScopedTimer<Tensor3<T>>(
                {{"r1upp", "r1low", "n", "r2"}, {r1u, r1l, n, r2}},                          // arguments
                {{r1u*n*r2*kernel_info::NoOp<T>()},                                          // flops
                 {r1u*n*r2*kernel_info::Load<T>() + (r1u+r1l)*n*r2*kernel_info::Store<T>()}} // data: load Up ; store C
            );

            C.resize(r1u + r1l, n, r2);
            
            for (int i2 = 0; i2 < r2; i2++)
            {
                for (int j = 0; j < n; j++)
                {
                    for (int i1 = 0; i1 < r1u; i1++)
                    {
                        C(i1, j, i2) = Up(i1, j, i2);
                    }
                    for (int i1 = 0; i1 < r1l; i1++)
                    {
                        C(i1 + r1u, j, i2) = (T)0;
                    }
                }
            }
        }

    } // namespace internal


    /**
     * @brief Add scaled tensor trains, where one of them is left-normalized.
     * 
     * Calculate gamma * y <- alpha * x + beta * y, such that for the result ||gamma * y|| = gamma
     * 
     * @warning Tensor x (TTx) must already be left-orthogonal.
     * 
     * @tparam T underlying data type (double, complex, ...)
     * 
     * @param alpha         coefficient of tensor x, scalar value
     * @param TTx           tensor x in tensor train format, left-orthogonal
     * @param beta          coefficient of tensor y, scalar value
     * @param TTy           tensor y in tensor train format
     * @param rankTolerance approxiamtion accuracy that is used to reduce the TTranks of the result
     * @param maxRank       maximal allowed TT-rank, enforced even if this violates the rankTolerance
     * @return              norm of the result tensor
     */
    template <typename T>
    T axpby_normalized(T alpha, const TensorTrain<T>& TTx_ortho, T beta, TensorTrain<T>& TTy, T rankTolerance = std::sqrt(std::numeric_limits<T>::epsilon()), int maxRank = 0x7fffffff)
    {
        const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();
        
        const auto& TTx = TTx_ortho;
        std::vector<Tensor3<T>>& y_cores = TTy.editableSubTensors();
        const std::vector<Tensor3<T>>& x_cores = TTx.subTensors();
        const std::vector<int>& x_dim = TTx.dimensions();
        const std::vector<int>& y_dim = TTy.dimensions();
        const int d = x_dim.size(); // order d

        assert(internal::is_normalized(TTx) == true);

        // check that dimensions match
        if (x_dim != y_dim)
            throw std::invalid_argument("TensorTrain axpby_normalized dimension mismatch!");
        
        if (x_cores[0].r1() != 1 || y_cores[0].r1() != 1 || x_cores[d-1].r2() != 1 || y_cores[d-1].r2() != 1)
            throw std::invalid_argument("TensorTrain axpby_normalized boundary ranks not equal to 1!");

        // special cases
        if (std::abs(alpha) == 0)
            return beta;
        
        if (std::abs(beta) == 0)
        {
            copy(TTx, TTy); // TTx -> TTy
            return alpha;
        }

        //
        // scale last tensor cores
        //
        Tensor3<T> x_last_core;
        copy(x_cores[d-1], x_last_core);
        internal::t3_scale(alpha, x_last_core);
        internal::t3_scale(beta, y_cores[d-1]);
        
        
        //
        // orthogonalization sweep left to right
        //

        // special case
        if (d == 1)
        {
            internal::t3_axpy((T)1, x_last_core, y_cores[0]);
        }

        //
        // Note: In the loop sweep, a few mem buffers could be reused -> giving better memory usage
        //

        Tensor3<T> Tyt;   // temporary 3Tensor y (Y tilde)
        Tensor3<T> Txt;   // temporary 3Tensor x (X tilde)
        Tensor2<T> Mzt;   // temporary 2Tensor to hold result (Txt(:,: x :)^tr * Tyt(:,: x :)

        Tensor2<T> Mtmp;  // temporary 2Tensor to hold result Tyt(:,: x :) - Txt(:,: x :) * Mzt(: x :)
        Tensor3<T> Ttmpu; // temporary 3Tensor to calculate upper part (in r1-dim) of Tyt into
        Tensor3<T> Ttmpl; // temporary 3Tensor to calculate lower part (in r1-dim) of Tyt into
        
        // initialize Txt and Tyt for k == 0
        copy(x_cores[0], Txt);
        copy(y_cores[0], Tyt);

        #ifdef VERBOSE
        printf("\n------------------ Before: -----------------\n\n");
        //printf("alpha: %f\n", alpha);
        //printf("beta:  %f\n", beta);
        printf("\nX:\n\n");
        internal::quickndirty_visualizeTT(TTx);
        printf("\nY:\n\n");
        internal::quickndirty_visualizeTT(TTy);

        printf("\n------------------ During: -----------------\n");
        #endif
        
        for (int k = 0; k < d - 1; k++)
        {
            // convenience references
            const Tensor3<T>& Ty1 = y_cores[k+1];
            const Tensor3<T>& Tx1 = (k == d-2) ? x_last_core : x_cores[k+1];

            //
            // Note: If Txt is square matrix once unfolded, most of the calculation (especially QR) can be skipped
            //

            // Mzt <- (Txt(:,: x :))^tr * Tyt(:,: x :)
            internal::zxtry(Txt, Tyt, Mzt);

            #ifdef VERBOSE
            printf("\n _____ Round %d _____:\n\n", k);
            printf("Matrix Z tilde:\n");
            internal::quickndirty_visualizeMAT(Mzt);
            #endif
            
            // Mtmp <- Tyt(:,: x :) - Txt(:,: x :) * Mzt(: x :)
            internal::t3232_fnmadd(Txt, Mzt, Tyt, Mtmp);

            #ifdef VERBOSE
            printf("Matrix Mtmp (taking QR of):\n");
            internal::quickndirty_visualizeMAT(Mtmp, 20);
            #endif

            // [Q, B] <- QR(Mtmp)
            const int c = Txt.r1();   // common r1-dimension of Txt and Tyt (= r_{k-1} + st_{k-1})
            const int n_k = Txt.n();  // n_k (n of Txt and Tyt)
            const int r_k = Txt.r2(); // r_k (r2 of Txt)
            const auto& [Q, B] = internal::m_normalize_qb(Mtmp, true, c*n_k - r_k);
            
            #ifdef VERBOSE
            printf("Matrix Q:\n");
            internal::quickndirty_visualizeMAT(Q);
            printf("Matrix B:\n");
            internal::quickndirty_visualizeMAT(B, 20);
            #endif

            // save result into y_cores[k] <- concat(Txt, Q, dim=3), unfolding Q: c*n_k x st_k -> c x n_k x st_k
            internal::t32_concat3(Txt, Q, y_cores[k]);
            
            #ifdef VERBOSE
            printf("result core %d:\n", k);
            internal::quickndirty_visualizeCORE(y_cores[k]);
            #endif

            //
            // Note: Ttmpu and Mzt can be independently calculated (in parallel)
            //

            // calculate upper half of new Tyt: Ttmpu <- Mzt *1 Ty1 (mode-1 contraction)
            internal::normalize_contract1(Mzt, Ty1, Ttmpu);

            // calculate lower half of new Tyt: Ttmpl <- B *1 Ty1
            internal::normalize_contract1(B, Ty1, Ttmpl);
            
            #ifdef VERBOSE
            printf("Ttmpl:\n");
            internal::quickndirty_visualizeCORE(Ttmpl);
            #endif

            // concatinate Tyt <- concat(Ttmpu, Ttmpl, dim=1)
            internal::t3_concat1(Ttmpu, Ttmpl, Tyt);
            
            #ifdef VERBOSE
            printf("Y tilde (of next round):\n");
            internal::quickndirty_visualizeCORE(Tyt);
            #endif

            //
            // Note: In Txt, a bunch of values are set to 0. Those could be left away, and the loops for Mzt, Mtmp, y_cores[k] updated accordingly (cuts on the flops there too)
            //
            
            // calculate new Txt: Txt <- concat(Tx1, 0, dim=1), 0 of dimension B.r1 x Tx1.n x Tx1.r2
            const int st_k = Q.r2();   // s^tilde_k (new s_k after QR)
            internal::t3_concat1_w0(Tx1, st_k, Txt);
            
            #ifdef VERBOSE
            printf("X tilde (of next round):\n");
            internal::quickndirty_visualizeCORE(Txt);
            printf("\n");
            #endif

        } // end loop

        // calculate y_cores[d-1] <- Txt + Tyt (componentwise)
        internal::t3_axpy((T)1, Txt, Tyt);
        std::swap(Tyt, y_cores[d-1]);

        #ifdef VERBOSE
        printf("\n----------------- After: -----------------\n\n");
        internal::quickndirty_visualizeTT(TTy);
        #endif

        //
        // compression sweep right to left
        //

        T gamma = rightNormalize(TTy, rankTolerance, maxRank);
        
        return gamma;
    }

}

#endif // PITTS_TENSORTRAIN_AXPBY_NORMALIZED_HPP