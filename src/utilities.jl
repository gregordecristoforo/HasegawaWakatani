export initial_condition, gaussian, sinusoidal, sinusoidalX, sinusoidalY, gaussianWallX, gaussianWallY

# ------------------------------- Initial fields -------------------------------------------

function gaussian(x, y; A=1, B=0, l=1)
    B + A * exp(-(x^2 + y^2) / (2 * l^2))
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

function randomIC(x, y)
    rand()
end

function initial_condition(fun::Function, domain::Domain; kwargs...)
    fun.(domain.x', domain.y; kwargs...)
end

# --------------------------- Analytical solutions -----------------------------------------

function HeatEquationAnalyticalSolution(prob)
    @. prob.u0 * exp(-prob.p["nu"] * prob.p["k2"] * prob.tspan[2])
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

#domain = Domain(1, 128, 1, 1)
#plot(burgers_equation_analytical_solution(domain, 0.155))

#nextprod((2,3), 90)

#P = plan_rfft(A,(2,1))

#maximum(real(P*A - rfft(A,2)))

#fftdims(P)

#using LinearAlgebra
#using LinearAlgebra: BlasReal

#A = rand(10)
#fft(A)
#k = fftfreq(10)

#[ for k in ks]

#sum([A[i]*exp(-im*2*π*(i-1)*0/length(A)) for i in eachindex(A)]) 

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

# Uses the Heat equation to test at the moment
function testTimestepConvergence(scheme, prob, analyticalSolution, timesteps)

    #Calculate analyticalSolution
    u = analyticalSolution(prob)

    #Initialize storage
    residuals = zeros(size(timesteps))

    for (i, dt) in enumerate(timesteps)
        #Change timestep of spectralODEProblem
        prob.dt = dt
        #Calculate approximate solution
        _, uN = scheme(prob, output=Nothing, singleStep=false)
        residuals[i] = norm(ifft(uN) - ifft(u))
    end

    #Plot residuals vs. time
    plot(timesteps, residuals, xaxis=:log, yaxis=:log, xlabel="dt", ylabel="||u-u_a||")
end

#
function testResolutionConvergence(scheme, prob, initialField, analyticalSolution, resolutions)
    cprob = deepcopy(prob)
    residuals = zeros(size(resolutions))

    for (i, N) in enumerate(resolutions)
        domain = Domain(N, 4)
        updateDomain!(cprob, domain)
        updateInitalField!(cprob, initialField)
        #prob = SpectralODEProblem(prob.f, domain, prob.u0, prob.tspan, p=prob.p, dt=prob.dt)
        #prob = SpectralODEProblem(prob.f, prob.domain, fft(initialField(prob.domain, prob.p)), prob.tspan, p = prob.p, dt=prob.dt)
        println(size(cprob.u0))
        u = analyticalSolution(cprob)

        _, uN = scheme(cprob, output=Nothing, singleStep=false)
        # Scaled residual to compensate for increased resolution
        residuals[i] = norm(ifft(u) - ifft(uN)) / (domain.Nx * domain.Ny)
        println(maximum(real(ifft(u))) - maximum(real(ifft(uN))))
    end

    display(plot(resolutions, residuals, xaxis=:log2, yaxis=:log))#, st=:scatter))
    display(plot(resolutions, resolutions .^ -2, xaxis=:log2, yaxis=:log))#, st=:scatter))
end

#testResolutionConvergence(mSS1Solve, prob, gaussianBlob, HeatEquationAnalyticalSolution, [16, 32, 64, 128, 256])

"""
"""
function HeatEquationAnalyticalSolution(n0, D, K, t)
    @. n0 * exp(D * K * t)
end

function compare(x, y, A::Matrix, B::Matrix)
    println(norm(A - B))
    #plot(x, y, A)
    #plot(x,x,B)
end

# Uses the Heat equation to test at the moment
function testResolutionConvergence(scheme, initialField, resolutions)
    D = 1.0
    dt = 0.001
    tend = 1

    residuals = zeros(size(resolutions))

    for (i, N) in enumerate(resolutions)
        domain = Domain(N, 4)
        k_x, k_y = getDomainFrequencies(domain)
        K = [-(k_x[i]^2 + k_y[j]^2) for i in eachindex(k_x), j in eachindex(k_y)]

        u0 = initialField.(domain.x, domain.y')
        analyticalSolution = HeatEquationAnalyticalSolution(u0, D, K, tend)

        du = similar(u0)

        #method(Laplacian, initialField)
        du = method(du, u0, K, dt, tend)
        #method(fun, t_span, dt, n0, p)
        # Scaled residual to compensate for increased resolution
        residuals[i] = norm(ifft(du) - ifft(analyticalSolution)) / (domain.Nx * domain.Ny)
    end

    plot(resolutions, residuals, xaxis=:log2, yaxis=:log, st=:scatter)
end