#!/usr/bin/env bash

# Copyright (c) 2023 German Aerospace Center (DLR), Institute for Software Technology, Germany
# SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
#
# SPDX-License-Identifier: BSD-3-Clause

# find mangled gcda files
GCDA_FILES=$(find $1 -name '#*.gcda')

# iterate over GCDA files
for gcda in $GCDA_FILES; do
  # unmangled name
  gcda_unmangled=$(echo $gcda | sed 's/#.*#//')
  gcno=$(echo $gcda | sed 's/.*\/#/#/' | sed 's/#/\//g' | sed 's/.gcda/.gcno/')
  gcda_dir=$(dirname $gcda)
  cp $gcno $gcda_dir
  mv $gcda $gcda_unmangled
done
