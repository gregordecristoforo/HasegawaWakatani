using FFTW
export Domain
include("spectralOperators.jl")

struct Domain
    Nx::Int64
    Ny::Int64
    Lx::Float64
    Ly::Float64
    dx::Float64
    dy::Float64
    x::LinRange
    y::LinRange
    kx::Frequencies
    ky::Frequencies
    SC::SpectralOperatorCoefficents
    Domain(N) = Domain(N, 1)
    Domain(N, L) = Domain(N, N, L, L)
    function Domain(Nx, Ny, Lx, Ly)
        dx = Lx / Nx
        dy = Ly / Ny
        x = LinRange(-Lx / 2, Lx / 2 - dx, Nx)
        y = LinRange(-Ly / 2, Ly / 2 - dy, Ny)
        kx = 2 * π * rfftfreq(Nx, 1 / dx)
        ky = 2 * π * fftfreq(Ny, 1 / dy)
        SC = SpectralOperatorCoefficents(kx, ky)
        new(Nx, Ny, Lx, Ly, dx, dy, x, y, kx, ky, SC)
    end
end



function getDomainFrequencies(domain::Domain)
    k_x = 2 * π * fftfreq(domain.Nx, 1 / domain.dx)
    k_y = 2 * π * fftfreq(domain.Ny, 1 / domain.dy)
    return k_x, k_y
end

function getRealDomainFrequencies(domain::Domain)
    k_x = 2 * π * rfftfreq(domain.Nx, 1 / domain.dx)
    k_y = 2 * π * fftfreq(domain.Ny, 1 / domain.dy)
    return k_x, k_y
end

struct SquareDomain
    Nx::Int64
    Ny::Int64
    Lx::Float64
    Ly::Float64
    dx::Float64
    dy::Float64
    x::LinRange
    y::LinRange
    kx::Frequencies
    ky::Frequencies
    SC::SpectralOperatorCoefficents
    SquareDomain(N) = SquareDomain(N, 1)
    SquareDomain(N, L) = SquareDomain(N, N, L, L)
    function SquareDomain(Nx, Ny, Lx, Ly)
        dx = Lx / Nx
        dy = Ly / Ny
        x = LinRange(-Lx / 2, Lx / 2 - dx, Nx)
        y = LinRange(-Ly / 2, Ly / 2 - dy, Ny)
        kx = 2 * π * rfftfreq(Nx, 1 / dx)
        ky = 2 * π * fftfreq(Ny, 1 / dy)
        println(typeof(kx))
        println(typeof(ky))
        SC = SpectralOperatorCoefficents(kx, ky)
        new(Nx, Ny, Lx, Ly, dx, dy, x, y, kx, ky, SC)
    end
end

using FFTW
# Assumed 1st direction uses rfft, while all others use fft

D = SquareDomain(32)

kappa = @. D.kx^2 + D.ky'^2
data = rand(32, 32)
uhat = rfft(data)

kappa .* uhat
D.kx .* uhat
D.ky' .* uhat

eltype(kappa)
length(kappa)
ndims(kappa)
size(kappa)
axes(kappa)
axes(kappa, 1)
axes(kappa, 2)
eachindex(kappa)
strides(kappa)

#Default is column vector
v = [1, 2]
A = [1 0; 0 3]
u = [1;; 2]
A * u

#Use comprehension
collect(1:4)
v = 1:2
B = reshape(collect(1:16), (2, 2, 2, 2))

function Diffusion(field, domain, nu)
    @. domain.SC.Laplacian * field
end

D = Domain(64, 2, 1, 1)

f = ones(64, 2)
fhat = rfft(f)

Diffusion(fhat, D, 0.1)