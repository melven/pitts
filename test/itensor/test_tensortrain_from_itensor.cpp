#include <gtest/gtest.h>
#include "pitts_tensortrain_from_itensor.hpp"
#include "pitts_tensortrain_axpby.hpp"
#include "pitts_tensortrain_dot.hpp"
#include "pitts_tensortrain_norm.hpp"
#include <itensor/all.h>
#include <iostream>


TEST(PITTS_TensorTrain_fromITensor, 2x2_unit)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  constexpr auto eps = 1.e-10;

  const int N = 2;
  const auto sites = itensor::SpinHalf(N,{"ConserveQNs=",false});

  //First set all spins to be "Up"
  auto state = itensor::InitState(sites,"Up");

  const auto mps_e11 = itensor::MPS(state);
  state.set(1,"Dn");
  const auto mps_e21 = itensor::MPS(state);
  state.set(2,"Dn");
  const auto mps_e22 = itensor::MPS(state);
  state.set(1,"Up");
  const auto mps_e12 = itensor::MPS(state);

  const TensorTrain_double tt_e11 = PITTS::fromITensor<double>(mps_e11);
  const TensorTrain_double tt_e21 = PITTS::fromITensor<double>(mps_e21);
  const TensorTrain_double tt_e12 = PITTS::fromITensor<double>(mps_e12);
  const TensorTrain_double tt_e22 = PITTS::fromITensor<double>(mps_e22);

  const std::vector<int> dimsRef = {2,2};
  ASSERT_EQ(dimsRef, tt_e11.dimensions());
  ASSERT_EQ(dimsRef, tt_e12.dimensions());
  ASSERT_EQ(dimsRef, tt_e21.dimensions());
  ASSERT_EQ(dimsRef, tt_e22.dimensions());

  const std::vector<int> ranksRef = {1};
  ASSERT_EQ(ranksRef, tt_e11.getTTranks());
  ASSERT_EQ(ranksRef, tt_e12.getTTranks());
  ASSERT_EQ(ranksRef, tt_e21.getTTranks());
  ASSERT_EQ(ranksRef, tt_e22.getTTranks());

  TensorTrain_double tt_ref(tt_e11.dimensions());

  tt_ref.setUnit({0,0});
  ASSERT_NEAR(0., axpby(-1., tt_e11, 1., tt_ref), eps);
  tt_ref.setUnit({1,0});
  ASSERT_NEAR(0., axpby(-1., tt_e21, 1., tt_ref), eps);
  tt_ref.setUnit({0,1});
  ASSERT_NEAR(0., axpby(-1., tt_e12, 1., tt_ref), eps);
  tt_ref.setUnit({1,1});
  ASSERT_NEAR(0., axpby(-1., tt_e22, 1., tt_ref), eps);
}


TEST(PITTS_TensorTrain_fromITensor, 2x2x2_unit)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  constexpr auto eps = 1.e-10;

  const int N = 3;
  const auto sites = itensor::SpinHalf(N,{"ConserveQNs=",false});

  //First set all spins to be "Up"
  auto state = itensor::InitState(sites,"Up");

  const auto mps_e111 = itensor::MPS(state);
  state.set(1,"Dn");
  const auto mps_e211 = itensor::MPS(state);
  state.set(1,"Up");
  state.set(2,"Dn");
  const auto mps_e121 = itensor::MPS(state);
  state.set(1,"Dn");
  const auto mps_e221 = itensor::MPS(state);
  state.set(1,"Up");
  state.set(2,"Up");
  state.set(3,"Dn");
  const auto mps_e112 = itensor::MPS(state);
  state.set(1,"Dn");
  const auto mps_e212 = itensor::MPS(state);
  state.set(1,"Up");
  state.set(2,"Dn");
  const auto mps_e122 = itensor::MPS(state);
  state.set(1,"Dn");
  const auto mps_e222 = itensor::MPS(state);

  const TensorTrain_double tt_e111 = PITTS::fromITensor<double>(mps_e111);
  const TensorTrain_double tt_e211 = PITTS::fromITensor<double>(mps_e211);
  const TensorTrain_double tt_e121 = PITTS::fromITensor<double>(mps_e121);
  const TensorTrain_double tt_e221 = PITTS::fromITensor<double>(mps_e221);
  const TensorTrain_double tt_e112 = PITTS::fromITensor<double>(mps_e112);
  const TensorTrain_double tt_e212 = PITTS::fromITensor<double>(mps_e212);
  const TensorTrain_double tt_e122 = PITTS::fromITensor<double>(mps_e122);
  const TensorTrain_double tt_e222 = PITTS::fromITensor<double>(mps_e222);

  const std::vector<int> dimsRef = {2,2,2};
  ASSERT_EQ(dimsRef, tt_e111.dimensions());
  ASSERT_EQ(dimsRef, tt_e211.dimensions());
  ASSERT_EQ(dimsRef, tt_e121.dimensions());
  ASSERT_EQ(dimsRef, tt_e221.dimensions());
  ASSERT_EQ(dimsRef, tt_e112.dimensions());
  ASSERT_EQ(dimsRef, tt_e212.dimensions());
  ASSERT_EQ(dimsRef, tt_e122.dimensions());
  ASSERT_EQ(dimsRef, tt_e222.dimensions());

  const std::vector<int> ranksRef = {1,1};
  ASSERT_EQ(ranksRef, tt_e111.getTTranks());
  ASSERT_EQ(ranksRef, tt_e211.getTTranks());
  ASSERT_EQ(ranksRef, tt_e121.getTTranks());
  ASSERT_EQ(ranksRef, tt_e221.getTTranks());
  ASSERT_EQ(ranksRef, tt_e112.getTTranks());
  ASSERT_EQ(ranksRef, tt_e212.getTTranks());
  ASSERT_EQ(ranksRef, tt_e122.getTTranks());
  ASSERT_EQ(ranksRef, tt_e222.getTTranks());

  TensorTrain_double tt_ref(tt_e111.dimensions());

  tt_ref.setUnit({0,0,0});
  ASSERT_NEAR(0., axpby(-1., tt_e111, 1., tt_ref), eps);
  tt_ref.setUnit({1,0,0});
  ASSERT_NEAR(0., axpby(-1., tt_e211, 1., tt_ref), eps);
  tt_ref.setUnit({0,1,0});
  ASSERT_NEAR(0., axpby(-1., tt_e121, 1., tt_ref), eps);
  tt_ref.setUnit({1,1,0});
  ASSERT_NEAR(0., axpby(-1., tt_e221, 1., tt_ref), eps);
  tt_ref.setUnit({0,0,1});
  ASSERT_NEAR(0., axpby(-1., tt_e112, 1., tt_ref), eps);
  tt_ref.setUnit({1,0,1});
  ASSERT_NEAR(0., axpby(-1., tt_e212, 1., tt_ref), eps);
  tt_ref.setUnit({0,1,1});
  ASSERT_NEAR(0., axpby(-1., tt_e122, 1., tt_ref), eps);
  tt_ref.setUnit({1,1,1});
  ASSERT_NEAR(0., axpby(-1., tt_e222, 1., tt_ref), eps);
}

TEST(PITTS_TensorTrain_fromITensor, 2x2_random)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  constexpr auto eps = 1.e-10;

  const int N = 2;
  const auto sites = itensor::SpinHalf(N,{"ConserveQNs=",false});

  const auto mps = itensor::randomMPS(sites, 2);
  std::cout << mps;

  const TensorTrain_double tt = PITTS::fromITensor<double>(mps);

  const std::vector<int> dimsRef = {2,2};
  ASSERT_EQ(dimsRef, tt.dimensions());

  const std::vector<int> ranksRef = {2};
  ASSERT_EQ(ranksRef, tt.getTTranks());


  //First set all spins to be "Up"
  auto state = itensor::InitState(sites,"Up");

  const auto mps_e11 = itensor::MPS(state);
  state.set(1,"Dn");
  const auto mps_e21 = itensor::MPS(state);
  state.set(2,"Dn");
  const auto mps_e22 = itensor::MPS(state);
  state.set(1,"Up");
  const auto mps_e12 = itensor::MPS(state);

  TensorTrain_double tt_e11(tt.dimensions());
  tt_e11.setUnit({0,0});
  TensorTrain_double tt_e21(tt.dimensions());
  tt_e21.setUnit({1,0});
  TensorTrain_double tt_e12(tt.dimensions());
  tt_e12.setUnit({0,1});
  TensorTrain_double tt_e22(tt.dimensions());
  tt_e22.setUnit({1,1});

  ASSERT_NEAR(inner(mps, mps_e11), dot(tt, tt_e11), eps);
  ASSERT_NEAR(inner(mps, mps_e21), dot(tt, tt_e21), eps);
  ASSERT_NEAR(inner(mps, mps_e12), dot(tt, tt_e12), eps);
  ASSERT_NEAR(inner(mps, mps_e22), dot(tt, tt_e22), eps);
}

TEST(PITTS_TensorTrain_fromITensor, 4x3x2_random)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  constexpr auto eps = 1.e-10;

  const int N = 3;
  auto sites = itensor::SiteSet(N);
  sites.set(1, itensor::GenericSite{itensor::Index{4}});
  sites.set(2, itensor::GenericSite{itensor::Index{3}});
  sites.set(3, itensor::GenericSite{itensor::Index{2}});

  const auto mps = itensor::randomMPS(sites, 2);
  std::cout << mps;

  const TensorTrain_double tt = PITTS::fromITensor<double>(mps);

  const std::vector<int> dimsRef = {4,3,2};
  ASSERT_EQ(dimsRef, tt.dimensions());

  const std::vector<int> ranksRef = {2,2};
  ASSERT_EQ(ranksRef, tt.getTTranks());

  ASSERT_NEAR(itensor::norm(mps), norm2(tt), eps);

  const auto other_mps = itensor::randomMPS(sites, 1);
  const TensorTrain_double other_tt = PITTS::fromITensor<double>(other_mps);

  ASSERT_NEAR(inner(mps, other_mps), dot(tt, other_tt), eps);
}
