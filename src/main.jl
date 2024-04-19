##
include("Operators.jl")
using .Operators
include("Timestepper.jl")
using .Timestepper
using FFTW
using Plots

##
using DifferentialEquations
using HDF5

# Read in input file
for line in readlines("input.txt")
    if line != ""
        if first(line) != "#"
            println(line)
        end
    end
end

## Gaussian example
function gaussianField(x, y, sx=1, sy=1)
    1 / (2 * π * sqrt(sx * sy)) * exp(-(x .^ 2 / sx + y .^ 2 / sy) / 2)
end

D = 1000000.0
N = 64

x = LinRange(-4, 4, N);
y = x;
a = fftfreq(N)

n0 = fft(gaussianField.(x, y', 1, 0.1))
tspan = (0.0, 200.0)
#problem = ODEProblem(Laplacian!, n0, tspan)
#sol = solve(problem)

for i in 1:10000
    n0 = timeStep(n0, a, 0, 0.01, Laplacian)
    if i % 100 == 1
        display(surface(x, y, real(ifft(n0)), zlims=(0, 0.2)))
    end
end