include("../src/Operators.jl")
using .Operators
using BenchmarkTools
using FFTW
using Einsum

## Testing of Laplacian operator

# Constants
N = 1024
x = LinRange(-1, 1, N);
y = x;

function gaussianField(x, y, sx=1, sy=1)
    1 / (2 * π * sqrt(sx * sy)) * exp(-(x .^ 2 / sx + y .^ 2 / sy) / 2)
end

# constant field
n0 = fft(gaussianField.(x, y', 1, 1))

## Using pre caluclated K = (k_x^2 + k_y^2)

"""
No pre allocation, but K = k_x^2 + k_y^2 is pre calculated
"""
function LaplacianPreKNOPreAllocForEach(n, K, t)
    return [n[i] * K[i] for i in eachindex(n)]
end

k_x = fftfreq(N)
k_y = fftfreq(N)
K = [-(k_x[i]^2 + k_y[j]^2) for i in eachindex(k_x), j in eachindex(k_y)]
n1 = similar(n0)

@btime LaplacianPreKNOPreAllocForEach(n1, K, 0)
#3.564 μs

## Comparing with einsum

# Using same K
LaplacianPreKNoPreAllocEinsum(n, K, t) = @einsum A[i] := K[i] * n[i]

@btime LaplacianPreKNoPreAllocEinsum(n1, K, 0)
# 1.020 μs

## Now trying with prealloc

function LaplacianPreKPreAllocNestedFor(r, n, K, t)
    r = Matrix{ComplexF64}([n[i, j] * K[i, j] for i in 1:size(n)[1], j in 1:size(n)[2]])
end

r = similar(n0)
n1 = deepcopy(n0)
@btime LaplacianPreKPreAllocNestedFor(r, n1, K, 0)
# 6.307 μs, still alocating, actually more

## Prealloc using einsum
LaplacianPreKPreAllocEinsum(r, n, K, t) = @einsum r[i] = K[i] * n[i]
r = similar(n0)
n1 = deepcopy(n0)
@btime LaplacianPreKPreAllocEinsum(r, n1, K, 0)
# 522.513 ns, no alloc DOES NOT YIELD RIGHT RESULTS HOWEVER

## Prealloc using einsum
LaplacianPreKPreAllocEinsumMatrix(r::Matrix{ComplexF64}, n::Matrix{ComplexF64}, K::Matrix{Float64}, t::Float64) = @einsum r[i, j] = K[i, j] * n[i, j]

r = similar(n0)
n1 = deepcopy(n0)
@btime LaplacianPreKPreAllocEinsumMatrix(r, n1, K, 0.0)
# 1.929 ms, no alloc

## Using built in elementwise multiplication
function LaplacianElementWise(r::Matrix{ComplexF64}, n::Matrix{ComplexF64}, K::Matrix{Float64}, t::Float64)
    @. r = n * K
end

r = similar(n0)
n1 = deepcopy(n0)
@btime LaplacianElementWise(r, n1, K, 0.0)
# 3.335 ms, 2 alloc
# 1.999 ms, 0 alloc using broadcasting

function rhs!(dn, n, p, t)
    Laplacian!(dn, n, p)
    @einsum dn[i, j] = 2 * dn[i, j]
end

function Laplacian!(dn, n, K)
    @einsum dn[i, j] = 2 * n[i, j] * K[i, j]
end

function Laplacian!(dn, n, K, D)
    @einsum dn[i, j] = D * n[i, j] * K[i, j]
end

@btime Laplacian!(r, n1, K, 2)

maxCFL, ind = findmax(real(n2))
println(ind[1], ind[2])