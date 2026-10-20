#!/bin/bash
# Copyright (c) 2026 German Aerospace Center (DLR), Institute for Software Technology, Germany
# SPDX-FileContributor: Claude Code (Anthropic AI model), prompted by Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Test of .claude/hooks/definition-heads.awk (the definition detection of the Stop hook
# stop-review-new-definitions.sh) on a sample with the expected and the excluded cases.

set -uo pipefail
cd "$(dirname "$0")/.."

expected='sample.hpp: template<typename T> void simple(int a, double b)
sample.hpp: T withDefaults(const Tensor3<T>& B, T tol = T(0),
sample.hpp: bool operator==(const Foo& other) const
sample.hpp: [[nodiscard]] int size() const
sample.hpp: class Foo final
sample.hpp: Foo(int a, const std::vector<int>& b) :
sample.hpp: static std::vector<int> limit(const std::vector<int>& dims)'

actual=$(awk -v file=sample.hpp -f .claude/hooks/definition-heads.awk <<'SAMPLE'
  template<typename T> void simple(int a, double b)
  {
    compute(a, b);
#pragma omp parallel for collapse(2)
    {
    }
    if( a > 0 )
    {
    }
    else
    {
    }
    do
    {
    } while(c);
    const auto lambda = [&](int i, double x)
    {
    };
    evaluate([&](int i)
    {
    });
    [&]<std::size_t... i>(std::index_sequence<i...>)
    {
    }(std::make_index_sequence<N>{});
  }

  template<typename T>
  T withDefaults(const Tensor3<T>& B, T tol = T(0),
                 int maxRank = std::numeric_limits<int>::max())
  {
  }

  enum class Kind
  {
  };

  bool operator==(const Foo& other) const
  {
  }

  [[nodiscard]] int size() const {return n_;}
  [[nodiscard]] int size() const
  {
  }

  class Foo final
  {
    public:
      Foo(int a, const std::vector<int>& b) :
        a_(a),
        b_(limit(b.empty() ?
          std::vector<int>(3, a) : b)),
        c_(std::min(a, 2))
      {
      }

      static std::vector<int> limit(const std::vector<int>& dims)
      {
        for(int k = 0; k < 3; k++)
        {
        }
        while( true )
        {
        }
        switch( k )
        {
        }
      }
  };
SAMPLE
)

if [ "$(printf '%s\n' "$actual" | sort)" != "$(printf '%s\n' "$expected" | sort)" ]; then
  echo "definition-heads.awk: unexpected result (< expected, > actual)"
  diff <(printf '%s\n' "$expected" | sort) <(printf '%s\n' "$actual" | sort)
  exit 1
fi
echo "definition-heads.awk: OK"
