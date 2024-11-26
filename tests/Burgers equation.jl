## Run all (alt+enter)
include("../src/HasagawaWakatini.jl")

# Run scheme test for Burgers equation
domain = Domain(1, 1024, 1, 20, realTransform=false, anti_aliased=false) #domain = Domain(64, 14)
u0 = initial_condition(gaussianWallY, domain)#, l=0.5)

plot(u0)

# Burgers equation 
function f(u, d, p, t)
    return -quadraticTerm(u, diffY(u, d), d)
end

# Parameters
parameters = Dict(
    "nu" => 0#0.01
)

# Break down time 
dudy = diffY(domain.transform.FT * u0, domain)
t_b = -1 / (minimum(real(domain.transform.iFT * dudy)))

# Time span
t_span = [0, 1.1 * t_b]

# Initialize problem
prob = SpectralODEProblem(f, domain, u0, t_span, p=parameters, dt=0.0001)

# Solve problem
using BenchmarkTools
@benchmark spectral_solve(prob, MSS3())

## ----------------------------------- Plot ------------------------------------------------

#Add analytical solution here
deepcopy(prob)

#u_anl = gaussianWall.(domain.x' - gaussianWall.(domain.x', domain.y)[1,:]'*t, domain.y)

testTimestepConvergence(mSS3Solve, prob, HeatEquationAnalyticalSolution, [0.1, 0.01, 0.001, 0.0001, 0.00001])

prob.dt = 0.001
testResolutionConvergence(mSS3Solve, prob, gaussianBlob, HeatEquationAnalyticalSolution, [16, 32, 64, 128, 256, 512, 1024])










































# ------------------------------------- Junk code ------------------------------------------

#updateDomain!(prob, domain)