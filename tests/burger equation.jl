include("../new src/domain.jl")
include("../new src/spectralODEProblem.jl")
include("../new src/schemes.jl")
include("../new src/utilities.jl")
include("../new src/quad.jl")

function gaussianWall(x, y, sx=1, sy=1)
    exp(-x .^ 2 / sx)
end

function f(u, p, t)
    du = im*Matrix([(p["kx"][j])*u[i,j] for i in eachindex(p["kx"]), j in eachindex(p["ky"])])
    -du#-1.5*quadraticTerm(u, du)
end

domain = Domain(64, 14)

using Plots

u0 = gaussianWall.(domain.x', domain.y, 5, 1)
plot(domain.x, domain.y, u0, st=:surface)
xlabel!("x")
ylabel!("y")

u_hat = fft(u0)

dt = 0.0001
parameters = Dict{String,Any}([("nu", 0)])
prob = SpectralODEProblem(f, domain, u_hat, [0, 1], p=parameters, dt=dt)

du_hat = f(u_hat, prob.p, 0)
ifftPlot(domain.x, domain.y, u, st=:surface)

t, u = mSS3Solve(prob, output=Nothing, singleStep=false)

ifftPlot(domain.x, domain.y, u, st=:surface)
xlabel!("x")
ylabel!("y")
title!("t = 3, without aliasing")
plot(domain.x, real(ifft(u))[1,:])

u_anl = gaussianWall.(domain.x' - gaussianWall.(domain.x', domain.y)[1,:]'*t, domain.y)

t = 1
u_anl = gaussianWall.(domain.x' - gaussianWall.(domain.x', domain.y)[1,:]'*t, domain.y)
plot(u_anl[1,:])