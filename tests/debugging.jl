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

#Use comprehension
collect(1:4)
v = 1:2
B = reshape(collect(1:16), (2, 2, 2, 2))

function Diffusion(field, domain, nu)
    @. domain.SC.Laplacian * field
end

D = Domain(64, 2, 1, 1)

f = ones(64, 2)
fhat = rfft(f)

Diffusion(fhat, D, 0.1)

# Old main code: 

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

A = [im, 1]
A'

##########################################

M = [(x,y) for y in 1:4, x in 1:6]
N = [(x,y) for x in 1:6, y in 1:4]


x = [1,2,3,4,5,6]
y = [1,2,3,4]
plot(x, y, M, st=:surface)
xlabel!("x")

mhat = rfft(M)

domain = Domain(6,4,1,1)

@. domain.ky*mhat

M*domain.ky

M = @. x'*x' + y

rfft(M)

h(x,y) = @. x + 0*y

j = h(x',y)
surface(x,y,j)
xlabel!("x")

rfft(j)

