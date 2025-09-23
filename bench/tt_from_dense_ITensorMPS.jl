# avoid some stupid recompilation!
#using Pkg
#Pkg.activate("./julia_startup_package/Startup")
#using Startup

using ITensors, ITensorMPS
ITensors.disable_warn_order()

n = parse(Int64, ARGS[1])
d = parse(Int64, ARGS[2])
r = parse(Int64, ARGS[3])

print("Generating random tensor of size ", n, "^", d, "\n")
print("Number of threads: ", Sys.CPU_THREADS, "\n")
print("rand wtime: ")
@time A = randn(n^d)
sites = siteinds(n,d)
print("calculate TT-SVD of rank ", r, "\n")
print("tt-svd wtime: ")
@time M = MPS(A,sites;cutoff=0.,maxdim=r)
print("tt-svd wtime: ")
@time M = MPS(A,sites;cutoff=0.,maxdim=r)
