/*! @file pitts_kernel_info.hpp
* @brief Helper types useful information on computational kernels (e.g. flops, memory operations)
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2020-04-24
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_KERNEL_INFO_HPP
#define PITTS_KERNEL_INFO_HPP

// includes
#include <type_traits>
#include <complex>


//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for helper functionality
  namespace internal
  {
    struct Flops final
    {
      bool noFMA;
      double singlePrecision;
      double doublePrecision;

    };

    constexpr Flops operator*(double n, const Flops& f) noexcept
    {
      return {f.noFMA, n*f.singlePrecision, n*f.doublePrecision};
    }

    constexpr Flops operator+(const Flops& a, const Flops& b) noexcept
    {
      return {a.noFMA || b.noFMA, a.singlePrecision + b.singlePrecision, a.doublePrecision + b.doublePrecision};
    }

    //!
    //!
    //! constexpr FlopsInfo flops = 20*Add<float> + 50*FMA<std::complex<double>>;
    //! => flops.noFMAs = true;
    //!    flops.singlePrecision = 20;
    //!    flops.doublePrecision = 200;
    //!
    //!
    template<bool B, int S, int D>
    struct BasicFlops
    {
      std::integral_constant<bool, B> noFMA;
      std::integral_constant<int,  S> singlePrecision;
      std::integral_constant<int,  D> doublePrecision;

      //! allow casting to Flops
      constexpr operator Flops() const noexcept
      {
        return Flops{noFMA, double(singlePrecision), double(doublePrecision)};
      }
    };


    template<typename> struct Add;
    template<typename> struct Mult;
    template<typename> struct FMA;


    template<> struct Add<float>                  : public BasicFlops<false, 1, 0> {};
    template<> struct Add<double>                 : public BasicFlops<false, 0, 1> {};
    template<> struct Add<std::complex<float>>    : public BasicFlops<false, 2, 0> {};
    template<> struct Add<std::complex<double>>   : public BasicFlops<false, 0, 2> {};
    template<> struct Mult<float>                 : public BasicFlops<false, 1, 0> {};
    template<> struct Mult<double>                : public BasicFlops<false, 0, 1> {};
    template<> struct Mult<std::complex<float>>   : public BasicFlops<false, 6, 0> {};
    template<> struct Mult<std::complex<double>>  : public BasicFlops<false, 0, 6> {};
    template<> struct FMA<float>                  : public BasicFlops<true, 2, 0> {};
    template<> struct FMA<double>                 : public BasicFlops<true, 0, 2> {};
    template<> struct FMA<std::complex<float>>    : public BasicFlops<true, 8, 0> {};
    template<> struct FMA<std::complex<double>>   : public BasicFlops<true, 0, 8> {};


/*
RooflineKernelData = collections.namedtuple('RooflineKernelData', ['singlePrecisionFlops',
                                                                   'doublePrecisionFlops',
                                                                   'noFMAs',
                                                                   'dataSize',
                                                                   'updateBytes',
                                                                   'loadBytes',
                                                                   'storeBytes'])
                                                                   */


  }
}


#endif // PITTS_KERNEL_INFO_HPP
