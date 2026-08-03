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
d = parse(Int64, ARGS[2])
rX = parse(Int64, ARGS[3])
rY = parse(Int64, ARGS[4])

print("Generating random MPS of size ", n, "^", d, " with bond dim. ", rX, " and with bond dim. ", rY, "\n")
#print("Number of threads: ", Sys.CPU_THREADS, "\n")
sites = siteinds(n,d)
print("rand X wtime: ")
@time X = random_mps(Float64, sites; linkdims=rX)
print("rand Y wtime: ")
@time Y = random_mps(Float64, sites; linkdims=rY)
print("right-orthogonalize X wtime: ")
@time orthogonalize!(X, 1)
print("right-orthogonalize Y wtime: ")
@time orthogonalize!(Y, 1)
for i in 1:4
  # print("truncated addition wtime: ")
  # @time Z = add(X, Y; maxdim=max(rX, rY))
  print("direct addition wtime: ")
  @time Z = add(X, Y; alg="directsum")
  print("truncation wtime: ")
  @time truncate!(Z; maxdim=max(rX, rY))
end
