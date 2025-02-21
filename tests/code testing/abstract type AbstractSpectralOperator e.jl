abstract type AbstractSpectralOperator end

# Create two types of operators
struct SpectralLinearOperator <: AbstractSpectralOperator 
    cache::Vector{AbstractArray}
    coefficient::Union{Number, AbstractArray}
    lastupdated::Number

    function SpectralLinearOperator(coefficient::Union{Number, AbstractArray}, dims=3)
        new(Vector{AbstractArray}(undef, dims), coefficient, 1)
    end
end

struct SpectralNonLinearOperator <: AbstractSpectralOperator end
    cache 
    TransformPlans
    lastupdated
end

function dx(u_hat::SubArray)
    println(parentindices(u_hat))
end

function (L::SpectralLinearOperator)(u::AbstractArray)
    L.cache = L.coefficient*u 
end

function (L::SpectralLinearOperator)(u::SubArray)
    cache[last(parentindices(u))] = coeff*u 
end

c = 1

L = SpectralLinearOperator(c)
u = zeros(12)

L(u)

#mat_update_func = (A, u, p, t) -> t * (p * p')

Base.Equivalent

d.quadraticTerm(dudx, dudy)

d.poissonBracket(u, v)


function Base.getindex(L::SpectralLinearOperator, i::Int)
    L.cache[i]
end

L(u)



#dndx, dndy
#dΩdx, dΩdy
#dϕdx, dϕdy 

function (poissonBracket)(out, u, v)
    out = d.quadraticTerm(d.dx(u),d.dx(v)) + d.quadraticTerm(d.dx(u),d.dy(v))
end

function N()
end
# Perhaps just use SciMLOperators?

function N(du, u, d, p, t)
    n, Ω, ϕ = eachslice(u)
    g, κ = p["g"], p["kappa"]
    ϕ = d.solvePhi(Ω)
    du[1] = -d.poissonBracket(u,v) + d.dy(ϕ) - g*d.dy(n) - σₙ*d.spectral_exp(ϕ)
    du[2] = -d.poissonBracket(u,v) - g*d.dy(n) + σₒ*(1 - d.spectral_exp(ϕ))
    return du
end

function N(du, u, d, p, t)
    n, Ω, ϕ = eachslice(u)
    g, σₙ, σₒ = p["g"], p["kappa"]
    ϕ = solvePhi(Ω,d)
    du[1] = -poissonBracket(u,v,d) + dy(ϕ,d) - g*dy(n,d) - σₙ*spectral_exp(ϕ,d)
    du[2] = -poissonBracket(u,v,d) - g*dy(n,d) + σₒ*(1 - d.spectral_exp(ϕ,d))
    return du
end

# Objectives: CUDA support, easy to write rhs without thinking too much about BTS, 
# pre-allocated operator results, if calculated once during timestep then use calculation

u0 = rand(64,64)

ic = [u0;;; u0;;; u0]
n, Ω, ϕ = eachslice(ic,dims=3)

function N(du, u, d, p, t)
    n, Ω, ϕ = eachslice(u)
    g, σₙ, σₒ = unpack(p)
    ϕ = solvePhi(Ω,d)
    du[1] = -poissonBracket(u,v,d) + dy(ϕ,d) - g*dy(n,d) - σₙ*spectral_exp(ϕ,d)
    du[2] = -poissonBracket(u,v,d) - g*dy(n,d) + σₒ*(1 - d.spectral_exp(ϕ,d))
    return du
end


struct testStruct
    c::AbstractArray 
end

domain = testStruct(CUDA.rand(Float64,1024,1024))

function spectral_exp!(du, ϕ)
    #@. du = d.c*ϕ
    broadcast!(*,du, ϕ, ϕ)
    return nothing
end

v = CUDA.rand(Float64, 1024,1024)
u = CUDA.rand(Float64, 1024,1024)

@time broadcast!(*, ru, rc, rv)

using CUDA
@CUDA.time spectral_exp!(u, v)#, domain)
@CUDA.profile spectral_exp!(u, v)

ru = rand(1024,1024)
rv = rand(1024,1024)
rc = rand(1024,1024)

using CUDA.CUFFT
using FFTW

kx = CUFFT.fftfreq(1024, 0.1)

@CUDA.time @. v = kx'*u

k = @views rv[:,1]

u = eachslice(u,dims=2)

function test(u,t)
    println(t)
end

u = rand(16,16)
t = 2

t = 200; nothing
f = u->(u,t)
test(f(u)...)