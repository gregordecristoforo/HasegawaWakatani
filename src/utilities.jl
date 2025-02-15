export initial_condition, gaussian, sinusoidal, sinusoidalX, sinusoidalY, gaussianWallX, gaussianWallY

# ------------------------------- Initial fields -------------------------------------------

function gaussian(x, y; A=1, B=0, l=1, x0=0, y0=0)
    B + A * exp(-((x - x0)^2 + (y - y0)^2) / (2 * l^2))
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

function randomIC(x, y)
    rand()
end

function initial_condition(fun::Function, domain::Domain; kwargs...)
    fun.(domain.x', domain.y; kwargs...)
end

function all_modes(domain::Domain, value=10^-6)
    n_hat = 1e-6 * ones(domain.Ny, domain.Nx)
    real(ifft(n_hat))
end

function all_modes_with_random_phase(domain::Domain, value=10^-6)
    θ = 2 * π * rand(domain.Ny, domain.Nx)
    n_hat = 1e-6 * ones(domain.Ny, domain.Nx) .* exp.(im * θ)
    n_hat[:, 1] .= 0
    n_hat[1, :] .= 0
    real(ifft(n_hat))
end

function initial_condition_linear_stability(domain::Domain, value=10^-6)
    θ = 2 * π * rand(domain.Ny, domain.Nx)
    phi_hat = 1e-6 * ones(domain.Ny, domain.Nx) .* exp.(im * θ)
    phi_hat[:, 1] .= 0
    phi_hat[1, :] .= 0
    n_hat = phi_hat .* exp(im * pi / 2)
    [real(ifft(n_hat));;; real(ifft(phi_hat))]
end

# --------------------------- Analytical solutions -----------------------------------------

function HeatEquationAnalyticalSolution(u0, domain, p, t)
    u0_hat = (domain.transform.FT * u0) .* exp.(p["nu"] * domain.SC.Laplacian * t)
    domain.transform.iFT * u0_hat
end

function HeatEquationAnalyticalSolution2(u0, domain, p, t)
    exp.(-(domain.x'.^2 .+ domain.y.^2)/(2*(1 + 2*p["nu"]*t)))/(1 + 2*p["nu"]*t)
end

# Burgers equation

# Part of "analytical" solution to Burgers equation with Gaussian waveform
function gaussian_diff_y(x, y; A=1, B=0, l=1)
    -A * y * exp(-(y^2) / (2 * l^2)) / (l^2)
end

function implicitBurgerSolution(u, x, t, f)
    u - f(x - u * t)
end

#implicitInviscidBurgerSolution.(0.1, 0, 1, y -> gaussian(0, y, l=0.08))

using Roots

function burgers_equation_analytical_solution(domain::Domain, t, f=y -> gaussian(0, y, l=0.08))
    u0 = initial_condition(gaussian, domain)
    [find_zero.(u -> implicitBurgerSolution(u, domain.y[yi], t, f), u0[yi, xi])
     for yi in eachindex(domain.y), xi in eachindex(domain.x)]
end

# ----------------------------- Inverse functions / transforms -----------------------------

function expTransform(u::AbstractArray)
    return [exp.(u[:, :, 1]);;; u[:, :, 2]]
end

# ------------------------------- Convergence testing --------------------------------------
function test_timestep_convergence(prob, analyticalSolution, timesteps, scheme=MSS3(), displayResults=true; kwargs...)

    #Calculate analyticalSolution
    u = analyticalSolution(prob.u0, prob.domain, prob.p, last(prob.tspan); kwargs...)

    #Initialize storage
    residuals = zeros(size(timesteps))

    for (i, dt) in enumerate(timesteps)
        # Create new spectralODEProblem with new dt
        newProb = SpectralODEProblem(prob.L, prob.f, prob.domain, prob.u0, prob.tspan, p=prob.p, dt=dt)

        output = Output(newProb, 2, [])

        #Calculate approximate solution
        sol = spectral_solve(newProb, scheme, output)
        residuals[i] = norm(sol.u[end] - u)
    end

    if displayResults
        # Plot residuals vs. time
        display(plot(timesteps, residuals, xaxis=:log, yaxis=:log, xlabel="dt", ylabel=L"||u-u_a||"))
    end

    return timesteps, residuals
end

#
function test_resolution_convergence(prob, initialField, analyticalSolution, resolutions,
    scheme=MMS3(), displayResults=true; kwargs...)

    od = prob.domain
    residuals = zeros(size(resolutions))

    for (i, N) in enumerate(resolutions)

        # Create higher resolution domain
        domain = Domain(N, N, od.Lx, od.Ly, realTransform=od.realTransform)#, anti_aliased = od.anti_aliased)
        u0 = initial_condition(initialField, domain) #TODO rethink initial condition 

        # Create new spectralODEProblem but with updated resolution
        newProb = SpectralODEProblem(prob.L, prob.f, domain, u0, prob.tspan, p=prob.p, dt=prob.dt)
        output = Output(newProb, 2, [])

        # Calculate solutions
        sol = spectral_solve(newProb, scheme, output)
        u = analyticalSolution(u0, domain, prob.p, last(prob.tspan); kwargs...)

        # Scaled residual to compensate for increased resolution
        residuals[i] = norm(sol.u[end] - u) / (domain.Nx * domain.Ny)
        println("N=$N, maximum error: $(argmax(sol.u[end] - u))")
    end

    if displayResults
        display(plot(resolutions, residuals, xaxis=:log2, yaxis=:log, xlabel=L"N_x \wedge N_y",
        marker = :circle, ylabel=L"||u-u_a||/N_xN_y"))
        display(plot!(resolutions, 1e-5*resolutions.^-2, linestyle=:dash))
        display(plot!(resolutions, 1e-5*resolutions.^-1, linestyle=:dash))
    end

    return resolutions, residuals
end

#plot(burgers_equation_analytical_solution(domain, 0.155))

# ------------------------------- Old "diagnostics" ----------------------------------------
#"""
#Checks if any of the many arguments that Plots.plot is complex 
#and if so takes the real part of the inverse Fourier transform.
#"""
"""
ifftPlot(args...; kwargs...)
Plot the real part of the inverse Fourier transform (IFFT) of each argument that is a complex array. 
This function is designed to handle multiple input arrays and plot them using the `plot` function 
from a plotting library such as Plots.jl. Non-complex arrays are plotted as-is.

# Arguments
- `args...`: A variable number of arguments. Each argument can be an array. If the array is of a complex type, 
  its IFFT is computed, and only the real part is plotted. If the array is not complex, it is plotted directly.
- `kwargs...`: Keyword arguments that are passed directly to the `plot` function to customize the plot.

# Usage
using FFTW, Plots

# Create some sample data
x = rand(ComplexF64, 100)\\
y = rand(100)

# Plot the real part of the IFFT of `x` and `y` directly
ifftPlot(x, y, title="IFFT Plot Example", legend=:topright)

"""
function ifftPlot(args...; kwargs...)
    processed_args = []
    for arg in args
        if eltype(arg) <: Complex
            arg = real(ifft(arg))
        end
        push!(processed_args, arg)
    end

    plot(processed_args...; kwargs...)
end

function compare(x, y, A::Matrix, B::Matrix)
    println(norm(A - B))
    #plot(x, y, A)
    #plot(x,x,B)
end