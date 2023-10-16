// Copyright (c) 2021 German Aerospace Center (DLR), Institute for Software Technology, Germany
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>
#include "pitts_hash_function.hpp"


TEST(PITTS_HashFunction, djb_hash)
{
  using PITTS::internal::djb_hash;
  EXPECT_EQ(djb_hash("hello"), djb_hash("hello"));
  EXPECT_NE(djb_hash("hello"), djb_hash("world"));
}


TEST(PITTS_HashFunction, djb_hash_at_compile_time)
{
  using PITTS::internal::djb_hash;
  static constexpr auto compileTimeHash = djb_hash("hello world");
  std::string hello_world = "hello world";
  const auto runTimeHash = djb_hash(hello_world);

  ASSERT_EQ(runTimeHash, compileTimeHash);
}

TEST(PITTS_HashFunction, djb_hash_combine_hashes)
{
  using PITTS::internal::djb_hash;
  ASSERT_EQ(djb_hash("world", djb_hash("hello ")), djb_hash("hello world"));
}
