# ------------------------------- Initial conditions ---------------------------------------

function gaussian(x, y; A=1, B=0, l=1, x0=0, y0=0)
    B + A * exp(-((x - x0)^2 + (y - y0)^2) / (2 * l^2))
end

function log_gaussian(x, y; A=1, B=1, l=1, x0=0, y0=0)
    log(gaussian(x, y; A=A, B=B, l=l, x0=x0, y0=y0))
end

function sinusoidal(x, y; Lx=1, Ly=1, n=1, m=1)
    sin(2 * π * n * x / Lx) * cos(2 * π * m * y / Ly)
end

function sinusoidalX(x, y; L=1, N=1)
    sin(2 * π * N * x / L)
end

function sinusoidalY(x, y; L=1, N=1)
    sin(2 * π * N * y / L)
end

function gaussianWallX(x, y; A=1, l=1)
    A * exp(-x^2 / (2 * l^2))
end

function gaussianWallY(x, y; A=1, l=1)
    A * exp(-y^2 / (2 * l^2))
end

function exponential_background(x, y; kappa=1)
    exp(-kappa * x)
end

function quadratic_function(x, y; kappa=1)
    abs(y) <= 1 ? 1 - y .^ 2 : 0.0
end

function randomIC(x, y)
    rand()
end

function initial_condition(fun::Function, domain::AbstractDomain; kwargs...)
    fun.(domain.x', domain.y; kwargs...)
end

function all_modes(domain::AbstractDomain, value=10^-6)
    n_hat = value * ones(domain.Ny, domain.Nx)
    real(ifft(n_hat))
end

function all_modes_with_random_phase(domain::AbstractDomain, value=10^-6)
    θ = 2 * π * rand(domain.Ny, domain.Nx)
    n_hat = value * ones(domain.Ny, domain.Nx) .* exp.(im * θ)
    n_hat[:, 1] .= 0
    n_hat[1, :] .= 0
    real(ifft(n_hat))
end

function initial_condition_linear_stability(domain::AbstractDomain, value=10^-6)
    θ = 2 * π * rand(domain.Ny, domain.Nx)
    phi_hat = value * ones(domain.Ny, domain.Nx) .* exp.(im * θ)
    #phi_hat[:, 1] .= 0
    #phi_hat[1, :] .= 0
    n_hat = phi_hat .* exp(im * pi / 2)
    [real(ifft(n_hat));;; real(ifft(phi_hat))]
end

# ---------------------- Inverse functions / transforms ------------------------------------

function expTransform(u::AbstractArray)
    return [exp.(u[:, :, 1]);;; u[:, :, 2]]
end

#------------------------------ Removal of modes -------------------------------------------
# TODO perhaps moved to utilitise

function remove_zonal_modes!(u::U, d::D) where {U<:AbstractArray,D<:AbstractDomain}
    @inbounds u[1, :, :] .= 0
end

function remove_streamer_modes!(u::U, d::D) where {U<:AbstractArray,D<:AbstractDomain}
    @inbounds u[:, 1, :] .= 0
end

function remove_asymmetric_modes!(u::U, domain::D) where {U<:AbstractArray,
    D<:AbstractDomain}
    if domain.Nx % 2 == 0
        @inbounds u[:, domain.Nx÷2+1, :] .= 0
    end
    if Ny % 2 == 0
        @inbounds u[domain.Ny÷2+1, :, :] .= 0
    end
end

function remove_nothing(u::U, d::D) where {U<:AbstractArray,D<:AbstractDomain}
    nothing
end

# ------------------------------------ Other -----------------------------------------------

# For parameter scans
function logspace(start, stop, length)
    10 .^ range(start, stop, length)
end

# TODO move to ext
# Extend plotting to allow domain as input
import Plots.plot
function plot(domain::AbstractDomain, args...; kwargs...)
    plot(domain.x, domain.y, args...; kwargs...)
end

# --------------------------------------- Mailing ------------------------------------------

function send_mail(subject; attachment="")
    if length(methods(send_mail)) == 1
        error("SMTPClient is not loaded. Please add SMTPClient.jl and configure the .env file.")
    else
        throw(MethodError(send_mail, Tuple{typeof(subject)}))
    end
end