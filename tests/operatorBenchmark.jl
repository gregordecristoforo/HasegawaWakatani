include("../src/Operators.jl")
using .Operators
using BenchmarkTools
using FFTW

## Testing of Laplacian operator

N = 64
x = LinRange(-1, 1, N);
y = x;
b = fftfreq(N)

function gaussianField(x, y, sx=1, sy=1)
    1 / (2 * π * sqrt(sx * sy)) * exp(-(x .^ 2 / sx + y .^ 2 / sy) / 2)
end

n0 = fft(gaussianField.(x, y', 1, 1))

function Laplacian(n, a, t)
    return Matrix{ComplexF64}([-100 * n[j, k] * (a[j]^2 + a[k]^2) for j in eachindex(a), k in eachindex(a)])
end

function LaplacianWithoutConverting(n, a, t)
    [n[j, k] * (a[j]^2 + a[k]^2) for j in eachindex(a), k in eachindex(a)]
end

a = [(b[j]^2 + b[k]^2) for j in eachindex(b), k in eachindex(b)]

using Einsum

n1 = similar(n0)
LaplacianUsingMeshgrid(n1, n, b, t) = @einsum n1[i, j] = n[i, j] * (b[i]^2 + b[j]^2)

n3 = @btime LaplacianUsingMeshgrid(n1, n0, b, 0)

b = fftfreq(N)
n2 = @btime LaplacianWithoutConverting(n0, b, 0)

using Plots

plot(x, y, real(ifft(n1)))
plot(x, y, real(ifft(n3)))