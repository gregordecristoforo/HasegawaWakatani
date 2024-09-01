using FFTW
export Domain

struct Domain
    Nx::Int64
    Ny::Int64
    Lx::Float64
    Ly::Float64
    dx::Float64
    dy::Float64
    x::LinRange
    y::LinRange
    Domain(N) = Domain(N, 1)
    Domain(N, L) = Domain(N, N, L, L)
    function Domain(Nx, Ny, Lx, Ly)
        dx = Lx / Nx
        dy = Ly / Ny
        x = LinRange(-Lx / 2, Lx / 2 - dx, Nx)
        y = LinRange(-Ly / 2, Ly / 2 - dy, Ny)
        new(Nx, Ny, Lx, Ly, dx, dy, x, y)
    end
end

function getDomainFrequencies(domain::Domain)
    k_x = 2 * π * fftfreq(domain.Nx, 1 / domain.dx)
    k_y = 2 * π * fftfreq(domain.Ny, 1 / domain.dy)
    return k_x, k_y
end