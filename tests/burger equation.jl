include("../new src/domain.jl")
include("../new src/spectralODEProblem.jl")
include("../new src/schemes.jl")
include("../new src/utilities.jl")
include("../new src/quad.jl")
include("../new src/spectralOperators.jl")
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

t, u = mSS3Solve(prob, output=Nothing, singleStep=false)

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