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

# Old main code: 

#Diffusion problem
function f(u, p, t)
    du = im * Matrix([(p["kx"][j]) * u[i, j] for i in eachindex(p["kx"]), j in eachindex(p["ky"])])
    -1.5 * quadraticTerm(u, du)#zero(u)#im*2*Matrix([(p["kx"][i] + p["ky"][j])*u[i,j] for i in eachindex(p["kx"]), j in eachindex(p["ky"])])
end

dt = 0.00001
parameters = Dict{String,Any}([("nu", 0.01)])
prob = SpectralODEProblem(f, domain, w0, [0, 3], p=parameters, dt=dt)

t, u = mSS3Solve(prob, output=Nothing, singleStep=false)

using Plots

plot(domain.x, domain.y, real(ifft(w0)), st=:surface)
plot(domain.x, domain.y, real(ifft(u)), st=:surface)
plot(domain.x, real(ifft(u))[1, :])
xlabel!("x")
ylabel!("y")

ifftPlot(domain.x, domain.y, u, title="Time step $(dt)", st=:surface)
ifftPlot(domain.x, domain.y, HeatEquationAnalyticalSolution(n0, 2, -prob.p["k2"], 0.1), title="Time step $(dt)", st=:surface)
ifftPlot(domain.x, domain.y, HeatEquationAnalyticalSolution(n0, 2, -prob.p["k2"], 0.1) - u)
##

function HeatEquationAnalyticalSolution(prob)
    @. prob.u0 * exp(-prob.p["nu"] * prob.p["k2"] * prob.tspan[2])
end


prob = SpectralODEProblem(f, domain, n0, [0, 0.1], p=parameters, dt=dt)
testTimestepConvergence(mSS3Solve, prob, HeatEquationAnalyticalSolution, [0.1, 0.01, 0.001, 0.0001, 0.00001])

prob.dt = 0.001
testResolutionConvergence(mSS3Solve, prob, gaussianBlob, HeatEquationAnalyticalSolution, [16, 32, 64, 128, 256, 512, 1024])

plot(domain.x, domain.y, real(ifft(HeatEquationAnalyticalSolution(prob))), st=:surface)

updateDomain!(prob, domain)

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