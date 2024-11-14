include("domain.jl")
include("spectralODEProblem.jl")
include("schemes.jl")
include("utilities.jl")

domain = Domain(64, 7)

function gaussianBlob(x, y, sx=1, sy=1)
    1 / (2 * π * sqrt(sx * sy)) * exp(-(x .^ 2 / sx + y .^ 2 / sy) / 2)
end

function gaussianWall(x, y, sx=1, sy=1)
    exp(-y .^ 2 / sx)
end

n0 = fft(gaussianBlob.(domain.x, domain.y', 0.01, 0.01))
w0 = fft(gaussianWall.(domain.x, domain.y'))

#Diffusion problem
function f(u, p, t)
    du = im*Matrix([(p["kx"][j])*u[i,j] for i in eachindex(p["kx"]), j in eachindex(p["ky"])])
    -1.5*quadraticTerm(u, du)#zero(u)#im*2*Matrix([(p["kx"][i] + p["ky"][j])*u[i,j] for i in eachindex(p["kx"]), j in eachindex(p["ky"])])
end

dt = 0.00001
parameters = Dict{String,Any}([("nu", 0.01)])
prob = SpectralODEProblem(f, domain, w0, [0, 3], p=parameters, dt=dt)

t, u = mSS3Solve(prob, output=Nothing, singleStep=false)

using Plots

plot(domain.x, domain.y, real(ifft(w0)), st=:surface)
plot(domain.x, domain.y, real(ifft(u)), st=:surface)
plot(domain.x, real(ifft(u))[1,:])
xlabel!("x")
ylabel!("y")

ifftPlot(domain.x, domain.y, u, title="Time step $(dt)", st=:surface)
ifftPlot(domain.x, domain.y, HeatEquationAnalyticalSolution(n0, 2, -prob.p["k2"], 0.1), title="Time step $(dt)", st=:surface)
ifftPlot(domain.x, domain.y, HeatEquationAnalyticalSolution(n0, 2, -prob.p["k2"], 0.1) - u)
##

include("diagnostics.jl")
function HeatEquationAnalyticalSolution(prob)
    @. prob.u0 * exp(-prob.p["nu"] * prob.p["k2"] * prob.tspan[2])
end

function gaussianBlob(domain, p)
    @. 1 / (2 * π * sqrt(p["sx"] * p["sy"])) * exp(-(domain.x^2 / p["sx"] + domain.y'^2 / p["sy"]) / 2)
end

parameters["sx"] = 0.1
parameters["sy"] = 0.1
domain = Domain(64)
n0 = fft(gaussianBlob(domain, parameters))

prob = SpectralODEProblem(f, domain, n0, [0, 0.1], p=parameters, dt=dt)
testTimestepConvergence(mSS3Solve, prob, HeatEquationAnalyticalSolution, [0.1, 0.01, 0.001, 0.0001, 0.00001])

prob.dt = 0.001
testResolutionConvergence(mSS3Solve, prob, gaussianBlob, HeatEquationAnalyticalSolution, [16, 32, 64, 128, 256, 512, 1024])

plot(domain.x, domain.y, real(ifft(HeatEquationAnalyticalSolution(prob))), st=:surface)

updateDomain!(prob, domain)

#t = LinRange(0, 1.23, 10^6)
#u = @. cos(20 * domain.x)
#u_hat = fft(u)
#u_hat[Int(domain.Nx / 2)+1] = 0
#un = forwardEuler(f, u_hat, t)
#plot(domain.x, real(ifft(un)))


##
#include("Operators.jl")
using .Operators
#include("Timestepper.jl")
using .Timestepper
using FFTW
using Plots

##
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


#Pseudocode
using SpectralSolve

#Define domain
SquareDomain(64)

#Define coefficent ODE
function f()
    "something"
end

#Define parameters

#Define initial condition

#Solve ODE from t0 to tend starting from initial condtion with these boundary condtions
#using the mSS3 algorithm. Save data every nth time, probe here
spectralSolve(f, "Something about time", u0, bc="periodic", alg="mSS3")

spectralSolve(prob, alg, output)


prob = SpectralODEProblem(f, domain, [0, 2], [2, 2])

spectralSolve(prob, mSS3)