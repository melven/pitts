#pragma once

#include "pitts_tensortrain.hpp"
#include "pitts_tensor2.hpp"
#include "pitts_tensor3_split.hpp"

namespace PITTS
{

    template <typename T>
    T _axpby_(T alpha, const TensorTrain<T>& TTx, T beta, TensorTrain<T>& TTy, 
        T rankTolerance = std::sqrt(std::numeric_limits<T>::epsilon()), int maxRank = 0x7fffffff);
}


namespace PITTS
{
    template <typename T>
    void unfold_left(const Tensor3<T>& T3, Tensor2<T>& T2)
    {
        const int r1 = T3.r1();
        const int n = T3.n();
        const int r2 = T3.r2();

        T2.resize(r1 * n, r2);

        // have to copy because memory layout is chunked up
        for (int i1 = 0; i1 < r1; i1++)
            for (int j = 0; j < n; j++)
                for (int i2 = 0; i2 < n; i2++)
                    T2(i1 + j*r1, i2) = T3(i1, j, i2);
    }

    template <typename T>
    void unfold_right(Tensor3<T> T3, Tensor2<T> T2)
    {
        const int r1 = T3.r1();
        const int n = T3.n();
        const int r2 = T3.r2();

        T2.resize(r1, n * r2);

        // have to copy because memory layout is chunked up
        for (int i1 = 0; i1 < r1; i1++)
            for (int j = 0; j < n; j++)
                for (int i2 = 0; i2 < n; i2++)
                    T2(i1, j + i2*n) = T3(i1,j,i2);
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
        
        // handle corner cases
        if( std::abs(alpha) == 0 )
            return beta;
        
        if( std::abs(beta) == 0 )
        {
            copy(TTx, TTy); // TTx -> TTy
            return alpha;
        }
        
        //
        // orthogonalization sweep left to right
        //

        Tensor3<T> Tyt;     // temporary 3Tensor y
        Tensor3<T> Txt;     // temporary 3Tensor x
        Tensor2<T> Mtemp;   // temporary 2Tensor to hold result (Txt(:,: x :)^tr * Tyt(:,: x :)
        Tensor2<T> Mtemp1;  // temporary 2Tensor to hold result Tyt(:,: x :) - Txt(:,: x :) * Mtemp(: x :)
        Tensor3<T> Ttempu;  // temporary 3Tensor to calculate upper part (in r1-dim) of Tyt into
        Tensor3<T> Ttempl;  // temporary 3Tensor to calculate lower part (in r1-dim) of Tyt into
        
        // initialize Txt and Tyt for k = 0
        copy(x_cores[0], Txt);
        copy(y_cores[0], Tyt);
        
        for (int k = 0; k < d - 1; k++)
        {
            // convenience references
            Tensor3<T>& Ty = y_cores[k];
            const Tensor3<T>& Ty1 = y_cores[k+1];
            const Tensor3<T>& Tx = x_cores[k];
            const Tensor3<T>& Tx1 = x_cores[k+1];

            // Mtemp <- (Txt(:,: x :)^tr * Tyt(:,: x :)
            {
                int r2x = Txt.r2();  // r2 of x
                int r2y = Tyt.r2();  // r2 of y
                int r1s = Txt.r1();  // shared dimension r1
                int ns = Txt.n();    // shared dimenion n

                Mtemp.resize(r2x, r2y);
                for (int j = 0; j < r2y; j++)
                {
                    for (int i = 0; i < r2x; i++)
                    {
                        Mtemp(i, j) = 0;
                        for (int k = 0; k < r1s; k++)
                        {
                            for (int l = 0; l < ns; l++)
                            {
                                Mtemp(i,j) += Txt(k, l, i) * Tyt(k, l, j);
                            }
                        }
                    }
                }
            }
            

            // Mtemp1 <- Tyt(:,: x :) - Txt(:,: x :) * Mtemp(: x :)
            {
                int r1s = Txt.r1();  // shared dimension r1
                int ns = Txt.n();    // shared dimenion n
                int r2y = Tyt.r2();  // r2 of y
                int r2x = Txt.r2();  // r2 of x

                Mtemp1.resize(r1s * ns, r2y);

                for (int j = 0; j < r2y; j++)
                {
                    for (int i2 = 0; i2 < ns; i2++)
                    {
                        for (int i1 = 0; i1 < r1s; i1++)
                        {
                            Mtemp1(i1 + i2*r1s, j) = Tyt(i1, i2, j);
                            for (int k = 0; k < r2x; k++)
                            {
                                Mtemp1(i1 + i2*r1s, j) -= Txt(i1, i2, k) * Mtemp(k, j);
                            }
                        }
                    }
                }
            }

            // [Q, B] <- QR(Mtemp1)
            const std::pair<Tensor2<T>,Tensor2<T>> [Q,B] = internal::normalize_qb(Mtemp1);

            // save result into y_cores[k] = Ty <- concat(Txt, B, dim=3)
            {
                int r1x = Txt.r1();
                int nx  = Txt.n();
                int r2x = Txt.r2();
                int r2Q = Q.r2();

                Ty.resize(r1x, nx, r2x + r2B);

                for (i2 = 0; i2 < r2x; i2++)
                {
                    for (int j = 0; j < nx; j++)
                    {
                        for (int i1 = 0; i1 < r1x; i1++)
                        {
                            Ty(i1, j, i2) = Txt(i1, j, i2);
                        }
                    }
                }
                for (int i2 = 0; i2 < r2Q; i2++)
                {
                    for (int j = 0; j < nx; j++)
                    {
                        for (int i1 = 0; i1 < r1x; i1++)
                        {
                            Ty(i1, j, i2 + r2x) = Q(i1 + j*r1x, i2);
                        }
                    }
                }
            }

            // calculate upper half of new Tyt: Ttempu <- Mtemp *1 Ty1
            //can be either in parallel with next one or next one can reuse Ttemp mem buffer
            {
                int r1 = Mtemp.r1();
                int s  = Ty1.r1();  // = Mtemp.r2();
                int n  = Ty1.n()
                int r2 = Ty1.r2();

                Ttempu.resize(r1, n, r2);

                for (int i2 = 0; i2 < r2; i2++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        for (int i1 = 0; i1 < r1; i1++)
                        {
                            Ttempu(i1, n, i2) = 0;
                            for (int k = 0; k < s; k++)
                            {
                                Ttempu(i1, n, i2) += Mtemp(i1, k) * Ty1(k, j, i2);
                            }
                        }
                    }
                }
            }

            // calculate upper half of new Tyt: Ttempl <- B *1 Ty1
            {
                int r1 = B.r1();
                int s  = Ty1.r1();  // = B.r2();
                int n  = Ty1.n()
                int r2 = Ty1.r2();

                Ttempl.resize(r1, n, r2);

                for (int i2 = 0; i2 < r2; i2++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        for (int i1 = 0; i1 < r1; i1++)
                        {
                            Ttempl(i1, n, i2) = 0;
                            for (int k = 0; k < s; k++)
                            {
                                Ttempl(i1, n, i2) += B(i1, k) * Ty1(k, j, i2);
                            }
                        }
                    }
                }
            }

            // concatinate Tyt <- concat(Ttempu, Ttempl, dim=1)
            {
                int r1u = Ttempu.r1();
                int r1l = Ttempl.r1();
                int ns = Ttempu.n();
                int r2s = Ttempu.r2();

                Tyt.resize(r1u + r1l, ns, r2s);
                
                for (int i2 = 0; i2 < r2s; i2++)
                {
                    for (int j = 0; j < ns; j++)
                    {
                        for (int i1 = 0; i1 < r1u; i1++)
                        {
                            Tyt(i1, j, i2) = Ttempu(i1, j, i2);
                        }
                        for (int i1 = 0; i1 < r1l; i1++)
                        {
                            Tyt(i1 + r1u, j, i2) = Ttempl(i1, j, i2);
                        }
                    }
                }
            }
            
            // calculate new Txt: Txt <- concat(Tx, 0, dim=1), 0 of dimension B.r1 x Tx.n x Tx.r2
            // set's a bunch of values to 0
            // those could be left away, and the loops that calculate Mtemp and Mtemp1 accordingly updated (cuts on the flops there too)
            {
                int r1u = Tx.r1();
                int r1l = B.r1(); // = Ttempl.r1()
                int ns = Tx.n();
                int r2s = Tx.r2();

                Txt.resize(r1u + r1l, ns, r2s);
                
                for (int i2 = 0; i2 < r2s; i2++)
                {
                    for (int j = 0; j < ns; j++)
                    {
                        for (int i1 = 0; i1 < r1u; i1++)
                        {
                            Txt(i1, j, i2) = Tx(i1, j, i2);
                        }
                        for (int i1 = 0; i1 < r1l; i1++)
                        {
                            Txt(i1 + r1u, j, i2) = 0;
                        }
                    }
                }
            }
        }

        // concatinate y_cores[d-1] <- concat(Txt, Tyt, dim=3)
        {
            int r1 = Txt.r1();
            int n = Txt.n();
            int r2x = Txt.r2();
            int r2y = Tyt.r2();

            y_cores[d-1].resize(r1x, nx, r2x + r2B);

            for (i2 = 0; i2 < r2x; i2++)
            {
                for (int j = 0; j < nx; j++)
                {
                    for (int i1 = 0; i1 < r1x; i1++)
                    {
                        y_cores[d-1](i1, j, i2) = Txt(i1, j, i2);
                    }
                }
            }
            for (int i2 = 0; i2 < r2Q; i2++)
            {
                for (int j = 0; j < nx; j++)
                {
                    for (int i1 = 0; i1 < r1x; i1++)
                    {
                        y_cores[d-1](i1, j, i2 + r2x) = Tyt(i1 + j*r1x, i2);
                    }
                }
            }
        }


        //
        // compression sweep right to left
        //

        T gamma = rightNormalize(TTy, rankTolerance, maxRank);
        
        return gamma;
    }




    template double _axpby_(double alpha, const TensorTrain<double>& TTx, double beta, TensorTrain<double>& TTy, double rankTolerance, int maxRank);
}