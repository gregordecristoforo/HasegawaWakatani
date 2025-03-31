## Run all (alt+enter)
include(relpath(pwd(), @__DIR__)*"/src/HasegawaWakatini.jl")

## Run scheme test for Burgers equation
domain = Domain(1, 1024, 1, 20, anti_aliased=true) 
u0 = initial_condition(gaussianWallY, domain)

# Diffusion 
function L(u, d, p, t)
    p["nu"] * diffusion(u, d)
end

# Burgers equation 
function N(u, d, p, t)
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
t_span = [0, 0.9 * t_b]

# Initialize problem
prob = SpectralODEProblem(L, N, domain, u0, t_span, p=parameters, dt=0.0001)

# Initialize output
output = Output(prob, 1000, [BurgerCFLDiagnostic(10), ProgressDiagnostic(10)], "burgers equation gaussian.h5")

## Solve problem
sol = spectral_solve(prob, MSS3(), output)
plot(sol.u[end])
#plot(sol.u[end] - burgers_equation_analytical_solution(u0, domain, parameters, last(t_span)))
#plot(burgers_equation_analytical_solution(u0, domain, parameters, last(t_span)))

## Time convergence test
timesteps = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
_, convergence1 = test_timestep_convergence(prob, burgers_equation_analytical_solution, timesteps, MSS1())
_, convergence2 = test_timestep_convergence(prob, burgers_equation_analytical_solution, timesteps, MSS2())
_, convergence3 = test_timestep_convergence(prob, burgers_equation_analytical_solution, timesteps, MSS3())
plot(timesteps, convergence1, xaxis=:log, yaxis=:log, label="MSS1")
plot!(timesteps, convergence2, xaxis=:log, yaxis=:log, label="MSS2", color="dark green")
plot!(timesteps, convergence3, xaxis=:log, yaxis=:log, label="MSS3", color="orange", xlabel="dt",
    ylabel=L"||U-u_a||", title="Timestep convergence, Burgers equation (N =$(domain.Ny))", xticks=timesteps)
savefig("Timestep convergence, Burgers equation (N =$(domain.Ny)).pdf")

## Resolution convergence test
resolutions = [2, 4, 8, 16, 32, 64, 128, 256]#, 512, 1024] #[2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
_, convergence1 = test_resolution_convergence(prob, gaussian, burgers_equation_analytical_solution, resolutions, MSS1())
_, convergence2 = test_resolution_convergence(prob, gaussian, burgers_equation_analytical_solution, resolutions, MSS2())
_, convergence3 = test_resolution_convergence(prob, gaussian, burgers_equation_analytical_solution, resolutions, MSS3())

plot(resolutions, convergence1, xaxis=:log2, yaxis=:log, label="MSS1")
plot!(resolutions, convergence2, xaxis=:log2, yaxis=:log, label="MSS2", color="dark green")
plot!(resolutions, convergence3, xaxis=:log2, yaxis=:log, label="MSS3", color="orange")
plot!(resolutions[1:end-4], 0.5 * exp.(-0.5 * resolutions)[1:end-4], label=L"\frac{1}{2}\exp\left(-\frac{N}{2}\right)", linestyle=:dash,
    xaxis=:log2, yaxis=:log, xticks=resolutions, xlabel=L"N_x \wedge N_y",
    ylabel=L"||U-u_a||/N_xN_y", title="Resolution convergence, Burgers equation (dt=$(prob.dt))")
savefig("Resolution convergence, Burgers equation (dt=$(prob.dt)).pdf")













extractOutput(out)
plot(out.u[end])

plot(out.diagnostics[3].data)
## ----------------------------------- Plot ------------------------------------------------

#Add analytical solution here
deepcopy(prob)

#u_anl = gaussianWall.(domain.x' - gaussianWall.(domain.x', domain.y)[1,:]'*t, domain.y)

testTimestepConvergence(mSS3Solve, prob, HeatEquationAnalyticalSolution, [0.1, 0.01, 0.001, 0.0001, 0.00001])

prob.dt = 0.001
testResolutionConvergence(mSS3Solve, prob, gaussianBlob, HeatEquationAnalyticalSolution, [16, 32, 64, 128, 256, 512, 1024])










































# ------------------------------------- Junk code ------------------------------------------

#updateDomain!(prob, domain)