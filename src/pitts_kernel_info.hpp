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
#include <cereal/cereal.hpp>


//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for helper functionality
  namespace internal
  {
    //! Helper data type for counting floating point operations of functions / kernels
    struct Flops final
    {
      //! Does this kernel/operation *not* allow FMA (fused multiply add) operations
      bool noFMA;

      //! number of single precision floating point operations
      double singlePrecision;

      //! number of double precision floating point operations
      double doublePrecision;


      //! allow reading/writing with cereal
      template<class Archive>
      void serialize(Archive & ar)
      {
        ar( CEREAL_NVP(noFMA),
            CEREAL_NVP(singlePrecision),
            CEREAL_NVP(doublePrecision) );
      }

      //! default equality operator
      constexpr bool operator==(const Flops&) const = default;
    };

    //! allow to multiply a Flops object with a scalar, e.g. the dimension of the data
    constexpr Flops operator*(double n, const Flops& f) noexcept
    {
      return {f.noFMA, n*f.singlePrecision, n*f.doublePrecision};
    }

    //! allow to add two Flops objects
    //!
    //! For simplitiy, this currently sets the noFMA flag when any of the two operations does not support FMA.
    //!
    constexpr Flops operator+(const Flops& a, const Flops& b) noexcept
    {
      return {a.noFMA || b.noFMA, a.singlePrecision + b.singlePrecision, a.doublePrecision + b.doublePrecision};
    }


    //! Helper type for pre-defining Flops of common operations depending on the data type
    //!
    //! constexpr Flops flops = 20*Add<float>() + 50*FMA<std::complex<double>>();
    //! => flops.noFMAs = true;
    //!    flops.singlePrecision = 20;
    //!    flops.doublePrecision = 400; // 8 * 50
    //!
    //!
    template<bool B, int S, int D>
    struct BasicFlops
    {
      //! Does this kernel/operation *not* allow FMA (fused multiply add) operations
      std::integral_constant<bool, B> noFMA;

      //! number of single precision floating point operations
      std::integral_constant<int,  S> singlePrecision;

      //! number of double precision floating point operations
      std::integral_constant<int,  D> doublePrecision;

      //! allow casting to Flops
      constexpr operator Flops() const noexcept
      {
        return Flops{noFMA, double(singlePrecision), double(doublePrecision)};
      }
    };


    //! Flops class for adding two numbers, requires an appropriate template specialization
    template<typename> struct Add;

    //! Flops class for multiplying two numbers, requires an appropriate template specialization
    template<typename> struct Mult;

    //! Flops class for a fused-multiply-add operation of two numbers, requires an appropriate template specialization
    template<typename> struct FMA;


    // template specializations for common numeric operations
    template<> struct Add<float>                  : public BasicFlops<true, 1, 0> {};
    template<> struct Add<double>                 : public BasicFlops<true, 0, 1> {};
    template<> struct Add<std::complex<float>>    : public BasicFlops<true, 2, 0> {};
    template<> struct Add<std::complex<double>>   : public BasicFlops<true, 0, 2> {};
    template<> struct Mult<float>                 : public BasicFlops<true, 1, 0> {};
    template<> struct Mult<double>                : public BasicFlops<true, 0, 1> {};
    template<> struct Mult<std::complex<float>>   : public BasicFlops<true, 6, 0> {};
    template<> struct Mult<std::complex<double>>  : public BasicFlops<true, 0, 6> {};
    template<> struct FMA<float>                  : public BasicFlops<false, 2, 0> {};
    template<> struct FMA<double>                 : public BasicFlops<false, 0, 2> {};
    template<> struct FMA<std::complex<float>>    : public BasicFlops<false, 8, 0> {};
    template<> struct FMA<std::complex<double>>   : public BasicFlops<false, 0, 8> {};

    // special type for parts that don't do any floating point operations at all
    template<typename> struct NoOp : public BasicFlops<false, 0, 0> {};


    //! helper class for counting data transfers
    struct Bytes final
    {
      //! total size of the data, used to check wether the data fits into some cache...
      double dataSize;

      //! number of bytes that are read and written
      double update;

      //! number of bytes that are only read
      double load;

      //! number of bytes that are only written
      double store;


      //! allow reading/writing with cereal
      template<class Archive>
      void serialize(Archive & ar)
      {
        ar( CEREAL_NVP(dataSize),
            CEREAL_NVP(update),
            CEREAL_NVP(load),
            CEREAL_NVP(store) );
      }

      //! default equality operator
      constexpr bool operator==(const Bytes&) const = default;
    };

    //! allow to multiply a Bytes object with a scalar, e.g. the dimension of the data
    constexpr Bytes operator*(double n, const Bytes& b) noexcept
    {
      return {n*b.dataSize, n*b.update, n*b.load, n*b.store};
    }

    //! allow to add two Bytes objects
    constexpr Bytes operator+(const Bytes& a, const Bytes& b) noexcept
    {
      return {a.dataSize+b.dataSize, a.update+b.update, a.load+b.load, a.store+b.store};
    }



    //! Helper type for pre-defining Bytes of common operations depending on the data type
    //!
    //! constexpr Bytes bytes = 20*Load<float>() + 50*Update<std::complex<double>>();
    //! => bytes.dataSize = 20*4 + 50*16;
    //!    bytes.update = 50*16;
    //!    bytes.load = 20*4;
    //!    bytes.store = 0;
    //!
    template<int D, int U, int L, int S>
    struct BasicBytes
    {
      //! total size of the data, used to check wether the data fits into some cache...
      std::integral_constant<int, D> dataSize;

      //! number of bytes that are read and written
      std::integral_constant<int, U> update;

      //! number of bytes that are only read
      std::integral_constant<int, L> load;

      //! number of bytes that are only written
      std::integral_constant<int, S> store;

      //! allow casting to Bytes
      constexpr operator Bytes() const noexcept
      {
        return Bytes{double(dataSize), double(update), double(load), double(store)};
      }
    };


    //! templates for common memory operations
    template<typename T> struct Update   : public BasicBytes<sizeof(T), sizeof(T), 0, 0> {};
    template<typename T> struct Load     : public BasicBytes<sizeof(T), 0, sizeof(T), 0> {};
    template<typename T> struct Store    : public BasicBytes<sizeof(T), 0, 0, sizeof(T)> {};
  }


  //! namespace for performance kernel information like floating point operations and data transfers
  namespace kernel_info
  {

    // for counting floating point operations
    using internal::Add;
    using internal::Mult;
    using internal::FMA;
    using internal::NoOp;

    // for counting data transfers
    using internal::Update;
    using internal::Load;
    using internal::Store;

    //! information for performance modeling of compute kernels, e.g. by a refined Roofline model
    struct KernelInfo final
    {
      //! number and type of floating point operations
      internal::Flops flops;

      //! number and kind of data transfers
      internal::Bytes bytes;


      //! allow reading/writing with cereal
      template<class Archive>
      void serialize(Archive & ar)
      {
        ar( CEREAL_NVP(flops),
            CEREAL_NVP(bytes) );
      }

      //! default equality operator
      constexpr bool operator==(const KernelInfo&) const = default;
    };
  }
}


#endif // PITTS_KERNEL_INFO_HPP
