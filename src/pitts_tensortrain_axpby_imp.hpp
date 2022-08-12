#pragma once

#include "pitts_tensortrain.hpp"
#include "pitts_tensortrain_norm.hpp"
#include "pitts_tensor2.hpp"
#include "pitts_tensor3_split.hpp"

//#define VERBOSE

namespace PITTS
{

    template <typename T>
    T _axpby_(T alpha, const TensorTrain<T>& TTx, T beta, TensorTrain<T>& TTy, 
        T rankTolerance = std::sqrt(std::numeric_limits<T>::epsilon()), int maxRank = 0x7fffffff);
}


namespace PITTS
{

            /*
            // get norm of M
            T nrm;// = internal::t2_nrm(M);
            for (int j = 0; j < M.r2(); j++)
                for (int i = 0; i < M.r1(); i++)
                    nrm += M(i, j) * M(i, j);
            nrm = std::sqrt(nrm);

            #ifdef VERBOSE
            std::cout << "* QR properties *\n";
            std::cout << "Norm of Mtemp: \t\t" << FIXED_FLOAT(nrm, 20) << std::endl;
            std::cout << "Mtmp == 0 tolerance: \t" << FIXED_FLOAT(rankTolerance, 20) << std::endl;     
            #endif

            // return empty matrices if M is approx 0
            if (nrm < rankTolerance)
            {
                #ifdef VERBOSE
                std::cout << "Skipping QR decomposition\n\n";
                #endif
                return result;
            }
            */

    namespace internal
    {
        #define FIXED_FLOAT(f, prec) std::fixed << std::setprecision(prec) << (f < 0 ? "" : " ") << f

        //! wrapper for qr, allows to show timings, returns LQ decomposition for leftOrthog=false
        template<typename T>
        auto m_normalize_qb(const Tensor2<T>& M, bool leftOrthog = true, int maxRank = std::numeric_limits<int>::max(), T rankTolerance = 0)
        {
            const auto timer = PITTS::timing::createScopedTimer<Tensor2<T>>();

            // get reasonable rank tolerance (ignoring passed value for now)        // was min(M.r1(), M.r2())
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


        /*
        * print the 2-tensor to command line (very ruggity)
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
        * print the 3-tensor to command line (very ruggity)
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
        * print the complete tensor to command line (very ruggity)
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


    }


    /**
     * @brief Add scaled tensor trains
     * 
     * Calculate gamma * y <- alpha * x + beta * y, such that for the result ||gamma * y|| = gamma
     * 
     * @warning Both tensors must already be left-orthogonal.
     * 
     * @tparam T underlying data type (double, complex, ...)
     * 
     * @param alpha         coefficient of tensor x, scalar value
     * @param TTx           tensor x in tensor train format, left-orthogonal
     * @param beta          coefficient of tensor y, scalar value
     * @param TTy           tensor y in tensor train format, left-orthogonal
     * @param rankTolerance approxiamtion accuracy that is used to reduce the TTranks of the result
     * @param maxRank       maximal allowed TT-rank, enforced even if this violates the rankTolerance
     * @return              norm of the result tensor
     */
    template <typename T>
    T _axpby_(T alpha, const TensorTrain<T>& TTx, T beta, TensorTrain<T>& TTy, T rankTolerance, int maxRank)
    {
        const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();

        std::vector<Tensor3<T>>& y_cores = TTy.editableSubTensors();
        const std::vector<Tensor3<T>>& x_cores = TTx.subTensors();
        const std::vector<int>& x_dim = TTx.dimensions();
        const std::vector<int>& y_dim = TTy.dimensions();
        const int d = x_dim.size(); // order d

        // check that dimensions match
        if( x_dim != y_dim )
            throw std::invalid_argument("TensorTrain axpby dimension mismatch!");
        
        // special cases
        if( std::abs(alpha) == 0 )
            return beta;
        
        if( std::abs(beta) == 0 )
        {
            copy(TTx, TTy); // TTx -> TTy
            return alpha;
        }

        // scale last tensor cores
        Tensor3<T> x_last_core;
        copy(x_cores[d-1], x_last_core);
        {
            Tensor3<T>& xc = x_last_core;
            const int xr1 = xc.r1();
            const int xn  = xc.n();
            for (int j = 0; j < xn; j++)
                for (int i1 = 0; i1 < xr1; i1++)
                    xc(i1, j, 0) *= alpha;

            Tensor3<T>& yc = y_cores[d-1];
            const int yr1 = yc.r1();
            const int yn  = yc.n();
            for (int j = 0; j < yn; j++)
                for (int i1 = 0; i1 < yr1; i1++)
                    yc(i1, j, 0) *= beta;
        }

        // special case
        if (d == 1)
        {
            add(y_cores[0], x_last_core, y_cores[0]);
            // then: calculate norm, scale by norm, return
        }
        
        
        //
        // orthogonalization sweep left to right
        //

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
        printf("alpha: %f\n", alpha);
        printf("beta:  %f\n\n", beta);
        printf("X:\n\n");
        internal::quickndirty_visualizeTT(TTx);
        printf("\nY:\n\n");
        internal::quickndirty_visualizeTT(TTy);

        printf("\n------------------ During: -----------------\n");
        #endif
        
        for (int k = 0; k < d - 1; k++)
        {
            // convenience references
            Tensor3<T>& Ty = y_cores[k];
            const Tensor3<T>& Ty1 = y_cores[k+1];
            const Tensor3<T>& Tx1 = (k == d-2) ? x_last_core : x_cores[k+1];

            const int c = Txt.r1();    // common r1-dimension of Txt and Tyt (= r_{k-1} + st_{k-1})
            const int n_k = Txt.n();   // n_k (n of Txt and Tyt)
            const int r_k = Txt.r2();  // r_k (r2 of Txt)
            const int s_k = Tyt.r2();  // s_k (r2 ot Tyt)
            const int n_k1 = Tx1.n();  // n_{k+1} (n of Txt and Tyt)
            const int r_k1 = Tx1.r2(); // r_{k+1} (r2 of Tx1)
            const int s_k1 = Ty1.r2(); // s_{k+1} (r2 of Ty1)

            //
            // Note: If Txt is square matrix once unfolded, most of the calculation (especially QR) can be skipped
            //

            // Mzt <- (Txt(:,: x :))^tr * Tyt(:,: x :)
            {
                Mzt.resize(r_k, s_k);

                for (int j = 0; j < s_k; j++)
                {
                    for (int i = 0; i < r_k; i++)
                    {
                        Mzt(i, j) = 0;
                        for (int l = 0; l < n_k; l++)
                        {
                            for (int k = 0; k < c; k++)
                            {
                                Mzt(i,j) += Txt(k, l, i) * Tyt(k, l, j);
                            }
                        }
                    }
                }
            }
            #ifdef VERBOSE
            printf("\n _____ Round %d _____:\n\n", k);
            printf("Matrix Z tilde:\n");
            internal::quickndirty_visualizeMAT(Mzt);
            #endif
            
            // Mtmp <- Tyt(:,: x :) - Txt(:,: x :) * Mzt(: x :)
            {
                Mtmp.resize(c * n_k, s_k);

                for (int j = 0; j < s_k; j++)
                {
                    for (int i2 = 0; i2 < n_k; i2++)
                    {
                        for (int i1 = 0; i1 < c; i1++)
                        {
                            Mtmp(i1 + i2*c, j) = Tyt(i1, i2, j);
                            for (int k = 0; k < r_k; k++)
                            {
                                Mtmp(i1 + i2*c, j) -= Txt(i1, i2, k) * Mzt(k, j);
                            }
                        }
                    }
                }
            }
            #ifdef VERBOSE
            printf("Matrix Mtmp (taking QR of):\n");
            internal::quickndirty_visualizeMAT(Mtmp, 20);
            #endif

            // [Q, B] <- QR(Mtmp)
            const auto& [Q, B] = internal::m_normalize_qb(Mtmp, true, c*n_k - r_k);

            const int st_k = Q.r2();   // s^tilde_k (new s_k after QR)

            #ifdef VERBOSE
            printf("Matrix Q:\n");
            internal::quickndirty_visualizeMAT(Q);
            printf("Matrix B:\n");
            internal::quickndirty_visualizeMAT(B, 20);
            #endif

            // save result into y_cores[k] = Ty <- concat(Txt, Q, dim=3), unfolding Q: c*n_k x st_k -> c x n_k x st_k
            {
                const int r1  = c;
                const int n   = n_k;
                const int r2r = r_k;
                const int r2l = st_k;

                Ty.resize(r1, n, r2r + r2l);

                for (int i2 = 0; i2 < r2r; i2++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        for (int i1 = 0; i1 < r1; i1++)
                        {
                            Ty(i1, j, i2) = Txt(i1, j, i2);
                        }
                    }
                }
                for (int i2 = 0; i2 < r2l; i2++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        for (int i1 = 0; i1 < r1; i1++)
                        {
                            Ty(i1, j, i2 + r2r) = Q(i1 + j*c, i2);
                        }
                    }
                }
            }
            #ifdef VERBOSE
            printf("result core %d:\n", k);
            internal::quickndirty_visualizeCORE(Ty);
            #endif

            //
            // Note: Ttmpu and Mzt can be independently calculated (in parallel)
            //

            // calculate upper half of new Tyt: Ttmpu <- Mzt *1 Ty1 (mode-1 contraction)
            {
                const int r1 = r_k;   // r1 of result (= #rows of matrix)
                const int s = s_k;    // shared dimension (= r1 of tensor = #cols of matrix)
                const int n = n_k1;   // n of result (= n of tensor)
                const int r2 = s_k1;  // r2 of result (= r2 of tensor)

                Ttmpu.resize(r1, n, r2);

                for (int i2 = 0; i2 < r2; i2++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        for (int i1 = 0; i1 < r1; i1++)
                        {
                            Ttmpu(i1, j, i2) = 0;
                            for (int k = 0; k < s; k++)
                            {
                                Ttmpu(i1, j, i2) += Mzt(i1, k) * Ty1(k, j, i2);
                            }
                        }
                    }
                }
            }

            // calculate lower half of new Tyt: Ttmpl <- B *1 Ty1
            {
                const int r1 = st_k;  // r1 of result (= #rows of matrix)
                const int s = s_k;    // shared dimension (= r1 of tensor = #cols of matrix)
                const int n = n_k1;   // n of result (= n of tensor)
                const int r2 = s_k1;  // r2 of result (= r2 of tensor)

                Ttmpl.resize(r1, n, r2);

                for (int i2 = 0; i2 < r2; i2++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        for (int i1 = 0; i1 < r1; i1++)
                        {
                            Ttmpl(i1, j, i2) = 0;
                            for (int k = 0; k < s; k++)
                            {
                                Ttmpl(i1, j, i2) += B(i1, k) * Ty1(k, j, i2);
                            }
                        }
                    }
                }
            }
            #ifdef VERBOSE
            printf("Ttmpl:\n");
            internal::quickndirty_visualizeCORE(Ttmpl);
            #endif

            // concatinate Tyt <- concat(Ttmpu, Ttmpl, dim=1)
            {
                const int r1u = r_k;
                const int r1l = st_k;
                const int n = n_k1;
                const int r2 = s_k1;

                Tyt.resize(r1u + r1l, n, r2);
                
                for (int i2 = 0; i2 < r2; i2++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        for (int i1 = 0; i1 < r1u; i1++)
                        {
                            Tyt(i1, j, i2) = Ttmpu(i1, j, i2);
                        }
                        for (int i1 = 0; i1 < r1l; i1++)
                        {
                            Tyt(i1 + r1u, j, i2) = Ttmpl(i1, j, i2);
                        }
                    }
                }
            }
            #ifdef VERBOSE
            printf("Y tilde (of next round):\n");
            internal::quickndirty_visualizeCORE(Tyt);
            #endif

            //
            // Note: In Txt, a bunch of values are set to 0. Those could be left away, and the loops for Mzt, Mtmp, y_cores[k] updated accordingly (cuts on the flops there too)
            //
            
            // calculate new Txt: Txt <- concat(Tx1, 0, dim=1), 0 of dimension B.r1 x Tx1.n x Tx1.r2
            {
                const int r1u = r_k;
                const int r1l = st_k;
                const int n = n_k1;
                const int r2 = r_k1;

                Txt.resize(r1u + r1l, n, r2);
                
                for (int i2 = 0; i2 < r2; i2++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        for (int i1 = 0; i1 < r1u; i1++)
                        {
                            Txt(i1, j, i2) = Tx1(i1, j, i2);
                        }
                        for (int i1 = 0; i1 < r1l; i1++)
                        {
                            Txt(i1 + r1u, j, i2) = 0;
                        }
                    }
                }
            }
            #ifdef VERBOSE
            printf("X tilde (of next round):\n");
            internal::quickndirty_visualizeCORE(Txt);
            printf("\n");
            #endif

        } // end loop

        // calculate y_cores[d-1] <- Txt + Tyt (componentwise)
        add(Txt, Tyt, y_cores[d-1]);

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




    template double _axpby_(double alpha, const TensorTrain<double>& TTx, double beta, TensorTrain<double>& TTy, double rankTolerance, int maxRank);
}