#!/usr/bin/bash

# abort on errors
set -xe

time -p source ~/load_modules_pitts_modules.sh > /dev/null

BUILD_DIR=$(mktemp -d -p . build.XXXX) && cd $BUILD_DIR

time -p cmake -Wno-dev -G Ninja -DCMAKE_BUILD_TYPE=$1 -DPITTS_EIGEN_USE_LAPACKE=On -DPITTS_USE_MODULES=Off -DCMAKE_CXX_COMPILER=clang++ -DPITTS_DEVELOP_BUILD=On ../../.. > cmake.log

time -p ninja > ninja1.log

time -p ninja pitts_tests > ninja2.log

touch ../../../src/pitts_tensortrain_dot_impl.hpp
time -p ninja > ninja3.log

touch ../../../src/pitts_tensortrain_dot_impl.hpp
time -p ninja pitts_tests > ninja4.log