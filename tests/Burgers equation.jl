## Run all (alt+enter)
using Plots
include("../src/domain.jl")
using .Domains
include("../src/diagnostics.jl")
include("../src/utilities.jl")
using LinearAlgebra
using LaTeXStrings
include("../src/spectralODEProblem.jl")
include("../src/schemes.jl")
include("../src/spectralSolve.jl")

## Run scheme test for Burgers equation
domain = Domain(1, 1024, 1, 20, realTransform=false, anti_aliased=false) #domain = Domain(64, 14)
u0 = initial_condition(gaussianWallY, domain)#, l=0.5)

plot(u0)

# Burgers equation 
function f(u, d, p, t)
    return -quadraticTerm(u, diffY(u, d), d)
end

# Other definition
function f2(u, d, p, t)
    return -diffY(quadraticTerm(u, u, d) / 2, d)
end

# Advection
function f3(u, d, p, t)
    return -100 * diffY(u, d)
end

# Parameters
parameters = Dict(
    "nu" => 0#0.01
)

# Break down time 
dudy = diffY(domain.transform.FT * u0, domain)
t_b = -1 / (minimum(real(domain.transform.iFT * dudy)))

t_span = [0, 1.1 * t_b]

prob = SpectralODEProblem(f2, domain, u0, t_span, p=parameters, dt=0.00001)

tend, uend = spectral_solve(prob, MSS3())

methods(quadraticTerm)

#plot(uend-u0)

## Solve and plot

plot!(real(domain.transform.iFT * dudy))

plot(initial_condition(gaussian_diff_y, domain))

plot(real(ifft(dudy)))

ifft_plan = plan_ifft(zero(u0))
plot(real(ifft_plan * dudy))


plot(domain.y, uend, xlabel="x", ylabel="y")

#Add analytical solution here

#u_anl = gaussianWall.(domain.x' - gaussianWall.(domain.x', domain.y)[1,:]'*t, domain.y)

testTimestepConvergence(mSS3Solve, prob, HeatEquationAnalyticalSolution, [0.1, 0.01, 0.001, 0.0001, 0.00001])

prob.dt = 0.001
testResolutionConvergence(mSS3Solve, prob, gaussianBlob, HeatEquationAnalyticalSolution, [16, 32, 64, 128, 256, 512, 1024])










































# ------------------------------------- Junk code ------------------------------------------

#updateDomain!(prob, domain)