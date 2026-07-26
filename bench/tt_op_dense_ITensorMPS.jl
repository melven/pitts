# Copyright (c) 2026 German Aerospace Center (DLR), Institute for Software Technology, Germany
# SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
#
# SPDX-License-Identifier: BSD-3-Clause


# avoid some stupid recompilation!
#using Pkg
#Pkg.activate("./julia_startup_package/Startup")
#using Startup

using ITensors, ITensorMPS
ITensors.disable_warn_order()
using ThreadPinning
openblas_pinthreads(:cores)
threadinfo(; blas=true, slurm=true)

n = parse(Int64, ARGS[1])
r = parse(Int64, ARGS[2])
rA = parse(Int64, ARGS[3])

print("Generating random MPO of size ", r, " x ", n, " x ", r, " with bond dim. ", rA, "\n")
#print("Number of threads: ", Sys.CPU_THREADS, "\n")
s = [Index(r, "rl"), Index(n, "n"), Index(r, "rr")]
l = [Index(rA, "l1"), Index(rA, "l1")]
print("rand A wtime: ")
@time A = MPO([
         random_itensor(Float64, s[1]', s[1], l[1]),
         random_itensor(Float64, l[1], s[2]', s[2], l[2]),
         random_itensor(Float64, l[2], s[3]', s[3]),
        ])
print("rand X wtime: ")
@time X = random_itensor(Float64, s...)
for i in 1:4
  print("apply MPO: ")
  @time Y = foldl(*, A; init=X)
  #print("Y:\n", Y)
end
