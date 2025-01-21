#Array functions
eltype(kappa)
length(kappa)
ndims(kappa)
size(kappa)
axes(kappa)
axes(kappa, 1)
axes(kappa, 2)
eachindex(kappa)
strides(kappa)

using Plots

ifftPlot(domain.x, domain.y, u, title="Time step $(dt)", st=:surface)
ifftPlot(domain.x, domain.y, HeatEquationAnalyticalSolution(n0, 2, -prob.p["k2"], 0.1), title="Time step $(dt)", st=:surface)
ifftPlot(domain.x, domain.y, HeatEquationAnalyticalSolution(n0, 2, -prob.p["k2"], 0.1) - u)
##

prob = SpectralODEProblem(f, domain, n0, [0, 0.1], p=parameters, dt=dt)
testTimestepConvergence(mSS3Solve, prob, HeatEquationAnalyticalSolution, [0.1, 0.01, 0.001, 0.0001, 0.00001])

prob.dt = 0.001
testResolutionConvergence(mSS3Solve, prob, gaussianBlob, HeatEquationAnalyticalSolution, [16, 32, 64, 128, 256, 512, 1024])

updateDomain!(prob, domain)

A[:, CartesianIndices(size(A)[end-1:end])]

using OrdinaryDiffEq

function lorenz!(du, u, p, t)
    du[1] = 10.0(u[2] - u[1])
    du[2] = u[1] * (28.0 - u[3]) - u[2]
    du[3] = u[1] * u[2] - (8 / 3) * u[3]
end
u0 = [[1.0; 0.0; 0.0];;; [1.0; 0.0; 0.0]]
tspan = (0.0, 100.0)
prob = ODEProblem(lorenz!, u0, tspan)

# Test that it worked
sol = solve(prob, Tsit5())
using Plots
plot(sol, vars=(1, 2, 3))

using Plots

sol.u
sol.t

size(sol.u[1])

size(u0)

v = Array{typeof(u0)}(10)

t = sol.t

u = map(eachindex(t)) do ti
    similar(u0)
end

using HDF5

A = Vector{Int}(1:10)
h5write("bar.h5", "fun", A .+ 1)

h5open("bar.h5", "w") do file
    g = create_group(file, "mygroup")
end

h5read("bar.h5", "fun")
h5writeattr("bar.h5", "fun", Dict("c" => "value for metadata parameter c", "d" => "metadata d"))

h5readattr("bar.h5", "fun")

include("../src/utilities.jl")
using .Domains


# Testing out 2D convolution
domain = Domain(64, 1)
u0 = initial_condition(gaussianWallY, domain, l=0.08)
du = domain.transform.iFT * diffY(domain.transform.FT * u0, domain)

using DSP

surface(domain.transform.iFT * quadraticTerm(domain.transform.FT * u0, diffY(domain.transform.FT * u0, domain), domain))
plotlyjsSurface(z=conv(domain.transform.FT * u0, domain.transform.FT * du))
surface(conv(du, u0))

surface(irfft(conv(domain.transform.FT * u0, domain.transform.FT * du), 128))

N = 100
x = 0:1:(N-1)
x = 2 * π * x ./ (N)
p = 100.01
A = [exp(im * p * x_j) for x_j in x]

plot(real(A))
