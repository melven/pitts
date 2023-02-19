#!/usr/bin/bash

# abort on errors
set -xe

time source ~/load_modules_pitts_modules.sh > /dev/null

BUILD_DIR=$(mktemp -d -p . build.XXXX) && cd $BUILD_DIR

time cmake -Wno-dev -G Ninja -DCMAKE_BUILD_TYPE=$1 -DPITTS_USE_MODULES=On -DCMAKE_CXX_COMPILER=clang++ ../../.. > cmake.log

time ninja > ninja1.log

time ninja pitts_tests > ninja2.log

touch ../../../src/pitts_tensortrain_dot.hpp
time ninja > ninja3.log

touch ../../../src/pitts_tensortrain_dot.hpp
time ninja pitts_tests > ninja4.log
