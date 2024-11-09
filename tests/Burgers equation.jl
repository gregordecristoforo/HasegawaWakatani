#include("../new src/domain.jl")
#include("../new src/spectralODEProblem.jl")
#include("../new src/schemes.jl")
#include("../new src/utilities.jl")
#include("../new src/quad.jl")
#include("../new src/spectralOperators.jl")
using Plots

function gaussianWallY(x, y, sx=1, sy=1)
    exp(-y .^ 2 / sx)
end

function gaussianWallX(x, y, sx=1, sy=1)
    exp(-x^2 / sx)
end

function f(u, d, p, t)
    #du = im*Matrix([(p["kx"][j])*u[i,j] for i in eachindex(p["kx"]), j in eachindex(p["ky"])])
    #-du#-1.5*quadraticTerm(u, du)
    DiffX(u, d.SC)
end


domain = Domain(64, 14)

u0 = gaussianWallX.(domain.x', domain.y, 5, 1)
plot(domain.x, domain.y, u0, st=:surface)
xlabel!("x")
ylabel!("y")

dt = 0.0001
parameters = Dict{String,Any}([("nu", 0)])
prob = SpectralODEProblem(f, domain, u0, [0, 1], p=parameters, dt=dt)

du_hat = f(prob.u0_hat, domain, prob.p, 0)
#ifftPlot(domain.x, domain.y, u, st=:surface)
plot(domain.x, domain.y, irfft(du_hat, domain.Nx)', st=:surface)

t, u = mSS1Solve(prob, output=Nothing, singleStep=false)

ifftPlot(domain.x, domain.y, u, st=:surface)
xlabel!("x")
ylabel!("y")
title!("t = 3, without aliasing")
plot(domain.x, real(ifft(u))[1, :])

#Add analytical solution here
t = 1
u_anl = gaussianWallX.(domain.x' - gaussianWallX.(domain.x', domain.y)[1, :]' * t, domain.y)

#u_anl = gaussianWall.(domain.x' - gaussianWall.(domain.x', domain.y)[1,:]'*t, domain.y)
plot(u_anl[1, :])

plot!(conj(rfft(u0')'[1, 1:end]), st=:scatter, ylims=(0,))
plot!(rfft(u0)[1, 1:end], st=:scatter)
plot(u0[1, 1:end])

plot(domain.x, u0[1, 1:end])

u_hat = rfft(u0)
du_hat = f(u_hat, domain, 0, 0)

plot(domain.x, domain.y, irfft(du_hat, domain.Ny), st=:surface)

domain.kx
domain.x
prob.u0_hat

#include("domain.jl")
#include("spectralODEProblem.jl")
#include("schemes.jl")
#include("utilities.jl")

domain = Domain(64)

function gaussianBlob(x, y, sx=1, sy=1)
    1 / (2 * π * sqrt(sx * sy)) * exp(-(x .^ 2 / sx + y .^ 2 / sy) / 2)
end

function gaussianWall(x, y, sx=1, sy=1)
    1 / (2 * π * sqrt(sx * sy)) * exp(-(x .^ 2 / sx + 0 * y .^ 2 / sy) / 2)
end

n0 = rfft(gaussianBlob.(domain.x, domain.y', 0.01, 0.01))

#Diffusion problem
function f(u, p, t)
    zero(u)#im*2*Matrix([(p["kx"][i] + p["ky"][j])*u[i,j] for i in eachindex(p["kx"]), j in eachindex(p["ky"])])
end

dt = 0.00001
parameters = Dict{String,Any}([("nu", 1)])
prob = SpectralODEProblem(f, domain, n0, [0, 0.1], p=parameters, dt=dt)

t, u_hat = mSS3Solve(prob, output=Nothing, singleStep=false)

using Plots

plot(domain.x, domain.y, real(irfft(u, 64)), st=:surface)

ifftPlot(domain.x, domain.y, u, title="Time step $(dt)", st=:surface)
ifftPlot(domain.x, domain.y, HeatEquationAnalyticalSolution(n0, 2, -prob.p["k2"], 0.1), title="Time step $(dt)", st=:surface)
ifftPlot(domain.x, domain.y, HeatEquationAnalyticalSolution(n0, 2, -prob.p["k2"], 0.1) - u)
##

#include("diagnostics.jl")
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
