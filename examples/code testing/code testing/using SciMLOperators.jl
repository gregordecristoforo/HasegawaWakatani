using SciMLOperators
using LinearAlgebra, FFTW
using CUDA

n = 256
L = 2π

dx = L / n
x = range(start=-L / 2, stop=L / 2 - dx, length=n) |> CuArray
u = @. sin(5x)cos(7x)
du = @. 5cos(5x)cos(7x) - 7sin(5x)sin(7x);
k = rfftfreq(n, 2π * n / L) |> CuArray
m = length(k)
P = plan_rfft(x)

fwd(u, p, t) = P * u
bwd(u, p, t) = P \ u

fwd(du, u, p, t) = mul!(du, P, u)
bwd(du, u, p, t) = ldiv!(du, P, u)

F = FunctionOperator(fwd, x, im * k;
    T=ComplexF64, op_adjoint=bwd,
    op_inverse=bwd,
    op_adjoint_inverse=fwd, islinear=true
)

ik = im * DiagonalOperator(k)
Dx = F \ ik * F

Dx = cache_operator(Dx, x)

@time ≈(Dx * u, du; atol=1e-8)
@time ≈(mul!(copy(u), Dx, u), du; atol=1e-8)

function N(du, u, d, p, t)
    n, Ω, ϕ = eachslice(u)
    g, σₙ, σₒ = unpack(p)
    ϕ = solvePhi(Ω, d)
    #^^ This is good start

    #vv This is the hard part
    #du[3] is not utilized because phi is not updated!
    du[1] = -poissonBracket(u, v, d) + diff_y(ϕ, d) - g * diff_y(n, d) - σₙ * spectral_exp(ϕ, d)
    du[2] = -poissonBracket(u, v, d) - g * diff_y(n, d) + σₒ * (1 - d.spectral_exp(ϕ, d))
    return du
end