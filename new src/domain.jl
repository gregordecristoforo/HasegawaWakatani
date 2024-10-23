using FFTW
export Domain
include("spectralOperators.jl")

# Assumed 1st direction uses rfft, while all others use fft
"""
Box domain, that calculates spatial resolution under construction.

# Contains
Lengths: ``Lx``, ``Ly`` (Float64)\\
Number of grid point: ``Nx``, ``Ny`` (Int64)\\
Spatial resolution: ``dx``, ``dy`` (Float64)\\
Spatial points: ``x``, ``y`` (LinRange)

``dxᵢ = 2Lₓ÷(Nₓ-1)``

Square Domain can be constructed using:\\
``Domain(N,L)``

Rectangular Domain can be constructed using:\\
``Domain(Nx,Ny,Lx,Ly)``
"""
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
    function Domain(Nx, Ny, Lx, Ly; real=true)
        dx = Lx / Nx
        dy = Ly / Ny
        x = LinRange(-Lx / 2, Lx / 2 - dx, Nx)
        y = LinRange(-Ly / 2, Ly / 2 - dy, Ny)
        # ------------------ If x-direction favored in rfft -------------------
        #kx = real ? 2 * π * rfftfreq(Nx, 1 / dx) : 2 * π * fftfreq(Nx, 1 / dx)
        #ky = 2 * π * fftfreq(Ny, 1 / dy)
        kx = 2 * π * fftfreq(Nx, 1 / dx)
        ky = real ? 2 * π * rfftfreq(Ny, 1 / dy) : 2 * π * fftfreq(Ny, 1 / dy)
        SC = SpectralOperatorCoefficents(kx, ky)
        new(Nx, Ny, Lx, Ly, dx, dy, x, y, kx, ky, SC)
    end
end