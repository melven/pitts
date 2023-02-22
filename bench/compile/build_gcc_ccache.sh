#!/usr/bin/bash

# abort on errors
set -xe

time source ~/load_modules_pitts.sh > /dev/null

BUILD_DIR=$(mktemp -d -p . build.XXXX) && cd $BUILD_DIR

time cmake -Wno-dev -G Ninja -DCMAKE_BUILD_TYPE=$1 -DPITTS_EIGEN_USE_LAPACKE=On -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DPITTS_USE_MODULES=Off -DCMAKE_CXX_COMPILER=g++ ../../.. > cmake.log

time ninja > ninja1.log

time ninja pitts_tests > ninja2.log

touch ../../../src/pitts_tensortrain_dot.hpp
time ninja > ninja3.log

touch ../../../src/pitts_tensortrain_dot.hpp
time ninja pitts_tests > ninja4.log
