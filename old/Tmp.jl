## Gaussian example
function gaussianField(x, y, sx=1, sy=1)
    1 / (2 * π * sqrt(sx * sy)) * exp(-(x .^ 2 / sx + y .^ 2 / sy) / 2)
end

##
include("Operators.jl")
using .Operators
include("Timestepper.jl")
using .Timestepper
include("Helperfunctions.jl")
using .Helperfunctions
using FFTW
using Plots
using BenchmarkTools

function HeatEquationAnalyticalSolution(n0, D, K, t)
    @. n0 * exp(D * K * t)
end

D = 1.0
domain = Domain(1024, 4)
k_x, k_y = getDomainFrequencies(domain)

# Adams–Moulton method
function timeStep!(du, u, A)
    @. du = A * u#[0]/(1 +  K*dt)
end

# Setting up tensors to do elementwise operations with
# First step
function firstStep!(u, u0, K, dt, tend)
    A1 = ones(size(u0))#./(1 .+  K.*dt)
    @. A1 /= (1 - K * dt)
    @views u1 = u#[1, :, :]

    u = deepcopy(u0)
    du = similar(u)
    for t in 0:dt:tend
        timeStep!(du, u, A1)
        u = deepcopy(du)
    end
    du
end

K = [-(k_x[i]^2 + k_y[j]^2) for i in eachindex(k_x), j in eachindex(k_y)]

## Run this cell
u0 = fft(gaussianField.(domain.x, domain.y', 1, 0.1))
dt = 0.001
tend = 1
du = similar(u0)

du = firstStep!(du, u0, K, dt, tend)
#plot(domain.x, domain.y, real(ifft(HeatEquationAnalyticalSolution(u0, 1, K, tend))) - real(ifft(du)))
ifftPlot(domain.x, domain.y, du, title="Time step $(dt)", st=:surface)
ifftPlot(domain.x, domain.y, HeatEquationAnalyticalSolution(u0, 1, K, tend) - du)
##

testTimestepConvergence(firstStep!, gaussianField, [0.1, 0.01, 0.001, 0.0001, 0.00001])
testResolutionConvergence(firstStep!, gaussianField, [16, 32, 64, 128, 256, 512, 1024])

##







sum(abs.(real(ifft(HeatEquationAnalyticalSolution(u0, 1, K, tend + 100000 * dt))) - real(ifft(du)))) / (Nx * Ny)

function secondStep!(u, u1, K, dt)
    A2 = zero(u1)
    @. A2 = (1 + 1 / 2 * dt * K) / (1 - 1 / 2 * dt * K)
    @views u2 = u[2, :, :]
    timeStep!(u2, u1, A2)
end

function thirdStep!(du, u12, K, dt)
    A3 = zero(u12)
    @. A3[1, :, :] = -(1 / 12) * dt * K / (1 - (5 / 12) * dt * K)
    @. A3[2, :, :] = (1 + (8 / 12) * dt * K) / (1 - (5 / 12) * dt * K)
    @views u34 = du[3:4, :, :]
    timeStep!(u34, u12, A3)
    du[3, :, :] = sum(u34, dims=1)
end

function AdamsMoulton3(u0, k_x, k_y, tspan, dt)
    K = [-(k_x[i]^2 + k_y[j]^2) for i in eachindex(k_x), j in eachindex(k_y)]
    ns = zeros(ComplexF64, 4, size(k_x)[1], size(k_y)[1])

    firstStep!(ns, n0, K, dt)
    @views n1 = ns[1, :, :]
    secondStep!(ns, n1, K, dt)
    @views n12 = ns[1:2, :, :]
    thirdStep!(ns, n12, K, dt)

    A = zeros(ComplexF64, size(ns[1:3, :, :]))
    @. A[1, :, :] = +1 / 24 * dt * K / (1 - (9 / 24) * dt * K)
    @. A[2, :, :] = -5 / 24 * dt * K / (1 - (9 / 24) * dt * K)
    @. A[3, :, :] = (1 + 19 / 24 * dt * K) / (1 - (9 / 24) * dt * K)

    B = similar(A)

    @views ns_new = ns[1:3, :, :]
    @views ns_old = ns[2:4, :, :]

    for t in tspan[1]+3*dt:dt:tspan[2]
        timeStep!(B, ns_new, A)
        ns[4, :, :] = sum(B, dims=1)
        #    @. ns_new = ns_old

        #display(plot(x,y,real(ifft(ns[4,:,:])), st=:surface, zlim=(0,0.4)))
    end
    plot(x, y, real(ifft(ns[4, :, :])) - real(ifft(HeatEquationAnalyticalSolution(n0, 1, K, 1))))
end

#@btime 
n0 = fft(gaussianField.(x, y', 1, 0.1))
AdamsMoulton3(n0, k_x, k_y, (0, 1), 0.01)

K = [-(k_x[i]^2 + k_y[j]^2) for i in eachindex(k_x), j in eachindex(k_y)]
plot(x, y, real(ifft(HeatEquationAnalyticalSolution(n0, 1, K, 1))))









using FFTW
using Plots

# Parameters
D = 1.0         # Diffusion coefficient
Lx = 10.0       # Domain size in x
Ly = 10.0       # Domain size in y
Nx = 64         # Number of grid points in x
Ny = 64         # Number of grid points in y
dt = 0.0001       # Time step
Nt = 10000      # Number of time steps

# Create grid
x = range(-4, stop=4, length=Nx)
y = range(-4, stop=4, length=Ny)

# Initial condition (e.g., Gaussian)
u0 = exp.(-((x .- Lx / 2) .^ 2 .+ (y' .- Ly / 2) .^ 2))

# Precompute wave numbers for spectral method
kx = 2 * pi / Lx * [0:Nx/2; -Nx/2+1:-1]
ky = 2 * pi / Ly * [0:Ny/2; -Ny/2+1:-1]
kx2 = kx .^ 2
ky2 = ky .^ 2

# Fourier transform of the initial condition
u_hat = deepcopy(n0)

# Function to compute the right-hand side of the spectral transformed equation
function rhs(u_hat, K, D)
    return D * K .* u_hat
end

for n = 1:Nt
    # Compute the right-hand side (RHS) for the current state
    k1 = rhs(u_hat, K, D)

    # First intermediate step
    u_hat1 = u_hat + 0.5 * dt * k1
    k2 = rhs(u_hat1, K, D)

    # Second intermediate step
    u_hat2 = u_hat - dt * k1 + 2 * dt * k2
    k3 = rhs(u_hat2, K, D)

    # Final update
    u_hat = u_hat + (dt / 6) * (k1 + 4 * k2 + k3)

    # Inverse Fourier transform to get back to the spatial domain
    u = ifft(u_hat)

    # Plotting or additional analysis can be done here
    # For example, to plot at specific time steps:
    if n % 100 == 0
        p = plot(x, y, real(u), title="Time step $(n)", xlabel="x", ylabel="y")
        display(p)
    end
end

plot(x, y, real(ifft(HeatEquationAnalyticalSolution(n0, 1, K, 1))))
plot(x, y, real(ifft(n0)), st=:surface)

