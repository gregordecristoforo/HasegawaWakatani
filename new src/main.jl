include("domain.jl")
include("spectralODEProblem.jl")
include("schemes.jl")
include("utilities.jl")

domain = Domain(64)

function gaussianBlob(x, y, sx=1, sy=1)
    1 / (2 * π * sqrt(sx * sy)) * exp(-(x .^ 2 / sx + y .^ 2 / sy) / 2)
end

function gaussianWall(x, y, sx=1, sy=1)
    1 / (2 * π * sqrt(sx * sy)) * exp(-(x .^ 2 / sx + 0*y .^ 2 / sy) / 2)
end

n0 = fft(gaussianBlob.(domain.x, domain.y', 0.01, 0.01))
w0 = fft(gaussianWall.(domain.x, domain.y', 0.001, 0.01))

function f(u, p, t)
    zero(u)#im*2*Matrix([(p["kx"][i] + p["ky"][j])*u[i,j] for i in eachindex(p["kx"]), j in eachindex(p["ky"])])
end

dt = 0.00001
parameters = Dict{String, Any}([("nu", 1)])
prob = SpectralODEProblem(f, domain, n0, [0, 0.1], p = parameters, dt=dt)

t, u = mSS3Solve(prob, output=Nothing, singleStep=false)


using Plots

plot(domain.x, domain.y, real(ifft(u)), st=:surface)

ifftPlot(domain.x, domain.y, u, title="Time step $(dt)", st=:surface)
ifftPlot(domain.x, domain.y, HeatEquationAnalyticalSolution(n0, 2, -prob.p["k2"], 0.1), title="Time step $(dt)", st=:surface)
ifftPlot(domain.x, domain.y, HeatEquationAnalyticalSolution(n0, 2, -prob.p["k2"], 0.1) - u)
##

include("diagnostics.jl")
function HeatEquationAnalyticalSolution(prob)
    @. prob.u0 * exp(-prob.p["nu"] * prob.p["k2"] * prob.tspan[2])
end

prob = SpectralODEProblem(f, domain, n0, [0, 1], p = parameters, dt=dt)
testTimestepConvergence(mSS3Solve, prob, HeatEquationAnalyticalSolution, [0.1, 0.01, 0.001, 0.0001, 0.00001])
testResolutionConvergence(mSS3Solve, prob, HeatEquationAnalyticalSolution, [16, 32, 64, 128, 256, 512, 1024])



updateDomain!(prob, domain)




#t = LinRange(0, 1.23, 10^6)
#u = @. cos(20 * domain.x)
#u_hat = fft(u)
#u_hat[Int(domain.Nx / 2)+1] = 0
#un = forwardEuler(f, u_hat, t)
#plot(domain.x, real(ifft(un)))





#Pseudocode
using SpectralSolve

#Define domain
SquareDomain(64)

#Define coefficent ODE
function f() "something" end

#Define parameters

#Define initial condition

#Solve ODE from t0 to tend starting from initial condtion with these boundary condtions
#using the mSS3 algorithm. Save data every nth time, probe here
spectralSolve(f, "Something about time", u0, bc="periodic", alg="mSS3")

spectralSolve(prob, alg, output)


prob = SpectralODEProblem(f, domain, [0,2], [2,2])

spectralSolve(prob, mSS3)
