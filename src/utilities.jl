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

# Part of "analytical" solution to Burgers equation with Gaussian waveform
function gaussian_diff_y(x, y; A=1, B=0, l=1)
    -A * y * exp(-(y^2) / (2 * l^2)) / (l^2)
end