/*! @file pitts_tensor3_unfold.hpp
* @brief Reshape a 3-dimensional tensor into a 2-dimensional tensor
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2022-06-27
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_TENSOR3_UNFOLD_HPP
#define PITTS_TENSOR3_UNFOLD_HPP

// includes
#include "pitts_tensor3.hpp"
#include "pitts_tensor2.hpp"
#include "pitts_multivector.hpp"
#include "pitts_performance.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! reshape a 3d tensor to a multivector combining the first two dimensions (r1*n x r2)
  //!
  //! @tparam T   underlying data type (double, complex, ...)
  //!
  //! @param t3   3d input tensor of dimension (r1,n,r2)
  //! @param mv   2d output tensor resized to dimension (r1*n,r2)
  //!
  template<typename T>
  void unfold_left(const Tensor3<T>& t3,  MultiVector<T>& mv)
  {
    const auto r1 = t3.r1();
    const auto n = t3.n();
    const auto r2 = t3.r2();

    const auto timer = PITTS::performance::createScopedTimer<Tensor3<T>>(
        {{"r1", "n", "r2"}, {r1, n, r2}},   // arguments
        {{r1*n*r2*kernel_info::NoOp<T>()},    // flops
         {r1*n*r2*kernel_info::Store<T>() + r1*n*r2*kernel_info::Load<T>()}}  // data
        );

    mv.resize(r1*n, r2);
#pragma omp parallel for collapse(3) schedule(static) if(r1*n*r2 > 500)
    for (int k = 0; k < r2; k++)
      for (int j = 0; j < n; j++)
        for (int i = 0; i < r1; i++)
        {
          mv(i+j*r1,k) = t3(i,j,k);
        }
  }

  //! reshape a 3d tensor to a mutlivector combining the last two dimensions (r1 x n*r2)
  //!
  //! @tparam T   underlying data type (double, complex, ...)
  //!
  //! @param t3   3d input tensor of dimension (r1,n,r2)
  //! @param mv   2d output tensor resized to dimension (r1,n*r2)
  //!
  template<typename T>
  void unfold_right(const Tensor3<T>& t3, MultiVector<T>& mv)
  {
    const auto r1 = t3.r1();
    const auto n = t3.n();
    const auto r2 = t3.r2();

    const auto timer = PITTS::performance::createScopedTimer<Tensor3<T>>(
        {{"r1", "n", "r2"}, {r1, n, r2}},   // arguments
        {{r1*n*r2*kernel_info::NoOp<T>()},    // flops
         {r1*n*r2*kernel_info::Store<T>() + r1*n*r2*kernel_info::Load<T>()}}  // data
        );

    mv.resize(r1, n*r2);
#pragma omp parallel for collapse(3) schedule(static) if(r1*n*r2 > 500)
    for (int k = 0; k < r2; k++)
      for (int j = 0; j < n; j++)
        for (int i = 0; i < r1; i++)
        {
          mv(i,j+k*n) = t3(i,j,k);
        }
  }

  //! reshape a 3d tensor to a vector flattening dimensions
  //!
  //! @tparam T             underlying data type (double, complex, ...)
  //!
  //! @param t3     3d input tensor of dimension (r1,n,r2)
  //! @param vec    1d output tensor resized to dimension (r1*n*r2)
  //!
  template<typename T>
  void unfold(const Tensor3<T>& t3, MultiVector<T>& mv)
  {
    const auto r1 = t3.r1();
    const auto n = t3.n();
    const auto r2 = t3.r2();

    const auto timer = PITTS::performance::createScopedTimer<Tensor3<T>>(
        {{"r1", "n", "r2"}, {r1, n, r2}},   // arguments
        {{r1*n*r2*kernel_info::NoOp<T>()},    // flops
         {r1*n*r2*kernel_info::Store<T>() + r1*n*r2*kernel_info::Load<T>()}}  // data
        );

    mv.resize(r1*n*r2, 1);
    const auto nChunks = mv.colStrideChunks();
#pragma omp parallel for schedule(static) if(nChunks > 50)
    for(long long iChunk = 0; iChunk < nChunks; iChunk++)
      mv.chunk(iChunk, 0) = t3.data()[iChunk];
  }


  //! reshape a 3d tensor into a 2d tensor, moving it's data
  //!
  //! @tparam T     underlying data type (double, complex, ...)
  //!
  //! @param t3     input tensor3 of dimension (r1,n,r2)
  //! @return       output tensor2 of dimension (r1,n*r2)
  //!
  template <typename T>
  Tensor2<T> unfold_right(Tensor3<T>&& t3)
  {
    // need to extract dimensions before move
    const auto reservedChunks = t3.reservedChunks();
    const auto r1 = t3.r1();
    const auto n = t3.n();
    const auto r2 = t3.r2();
    std::unique_ptr<Chunk<T>[]> data = std::move(t3);
    return Tensor2<T>(std::move(data), reservedChunks, r1, n*r2);
  }

  //! reshape a 3d tensor into a 2d tensor, moving it's data
  //!
  //! @tparam T     underlying data type (double, complex, ...)
  //!
  //! @param t3     input tensor3 of dimension (r1,n,r2)
  //! @return       output tensor2 of dimension (r1*n,r2)
  //!
  template <typename T>
  Tensor2<T> unfold_left(Tensor3<T>&& t3)
  {
    // need to extract dimensions before move
    const auto reservedChunks = t3.reservedChunks();
    const auto r1 = t3.r1();
    const auto n = t3.n();
    const auto r2 = t3.r2();
    std::unique_ptr<Chunk<T>[]> data = std::move(t3);
    return Tensor2<T>(std::move(data), reservedChunks, r1*n, r2);
  }
  
  //! reshape a 3d tensor to a vector flattening dimensions (without copying data)
  //!
  //! @tparam T             underlying data type (double, complex, ...)
  //!
  //! @param t3     3d input tensor of dimension (r1,n,r2), moved from
  //! @return       1d output tensor resized to dimension (r1*n*r2)
  //!
  template<typename T>
  MultiVector<T> unfold(Tensor3<T>&& t3)
  {
    // need to extract dimensions before move
    const auto reservedChunks = t3.reservedChunks();
    const auto r1 = t3.r1();
    const auto n = t3.n();
    const auto r2 = t3.r2();
    std::unique_ptr<Chunk<T>[]> data = std::move(t3);
    return MultiVector<T>(std::move(data), reservedChunks, r1*n*r2, 1);
  }


  //! create a Tensor2 view of a right-unfolded Tensor3
  //!
  //! @tparam T     underlying data type (double, complex, ...)
  //!
  //! @param t3     [in]  Tensor3 of dimension (r1,n,r2)
  //! @param t2     [out] Tensor2 alias of dimension (r1,n*r2)
  //!
  template<typename T>
  Tensor2View<T> unfold_right(Tensor3<T>& t3)
  {
    return Tensor2View<T>(t3.data(), t3.r1(), t3.n()*t3.r2());
  }

  template<typename T>
  ConstTensor2View<T> unfold_right(const Tensor3<T>& t3)
  {
    return ConstTensor2View<T>(const_cast<Chunk<T>*>(t3.data()), t3.r1(), t3.n()*t3.r2());
  }

  //! create a Tensor2 view of a left-unfolded Tensor3
  //!
  //! @tparam T     underlying data type (double, complex, ...)
  //!
  //! @param t3     [in]  Tensor3 of dimension (r1,n,r2)
  //! @param t2     [out] Tensor2 alias of dimension (r1*n,r2)
  //!
  template<typename T>
  Tensor2View<T> unfold_left(Tensor3<T>& t3)
  {
    return Tensor2View<T>(t3.data(), t3.r1()*t3.n(), t3.r2());
  }

  template<typename T>
  ConstTensor2View<T> unfold_left(const Tensor3<T>& t3)
  {
    return ConstTensor2View<T>(const_cast<Chunk<T>*>(t3.data()), t3.r1()*t3.n(), t3.r2());
  }
}
  
#endif // PITTS_TENSOR3_UNFOLD_HPP
