include("../new src/domain.jl")
include("../new src/spectralODEProblem.jl")
include("../new src/schemes.jl")
include("../new src/utilities.jl")
include("../new src/quad.jl")
include("../new src/spectralOperators.jl")

function gaussianWall(x, y, sx=1, sy=1)
    exp(-y .^ 2 / sx)
end

function gaussianWall(X, Y, sx=1, sy=1)
    [exp(-x .^ 2 / sx) for x in X, y in Y]
end

function f(u, d, p, t)
    #du = im*Matrix([(p["kx"][j])*u[i,j] for i in eachindex(p["kx"]), j in eachindex(p["ky"])])
    #-du#-1.5*quadraticTerm(u, du)
    DiffX(u, d.SC)
end

domain = Domain(4, 14)

using Plots

u0 = gaussianWall.(domain.x, domain.y, 5, 1)
plot(domain.x, domain.y, u0, st=:surface)
xlabel!("x")
ylabel!("y")

u_hat = rfft(u0)



dt = 0.0001
parameters = Dict{String,Any}([("nu", 0)])
prob = SpectralODEProblem(f, domain, u_hat, [0, 1], p=parameters, dt=dt)

du_hat = @. u_hat*domain.ky'

du_hat = f(u_hat, domain, prob.p, 0)
#ifftPlot(domain.x, domain.y, u, st=:surface)
plot(domain.x, domain.y, irfft(du_hat, domain.Nx), st=:surface)

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

##########################################

M = [y for y in 1:4, x in 1:4]

x = 1:4
y = 1:4
plot(x, y, M, st=:surface)
xlabel!("x")

mhat = rfft(M)

