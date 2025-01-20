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