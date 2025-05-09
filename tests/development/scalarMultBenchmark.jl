using BenchmarkTools
using Einsum

M = rand(ComplexF64, 3, 1024,1024)
N = similar(M)
a = [1, 2, 3] 

function scalarMult!(N, M, a)
    @. N = a*M
end

@btime scalarMult!(N, M, a)
#5.831 ms 0 alloc

N2

scalarMultEinsum(N,M,a) = @einsum N[i,j,k] = a[i]*M[i,j,k]

@btime scalarMultEinsum(similar(M), M, a)
#5.936 ms