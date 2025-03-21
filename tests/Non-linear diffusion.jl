## Run all (alt+enter)
include("../src/HasegawaWakatini.jl")

## Run scheme test for Burgers equation
domain = Domain(1024, 1024, 50, 50, anti_aliased=false)
u0 = log.(initial_condition(gaussian, domain; B = 1))

# Diffusion 
function L(u, d, p, t)
    p["nu"]*diffusion(u, d)
end

function N(u, d, p, t)
    p["nu"]*(quadraticTerm(diffX(u, d), diffX(u, d),d) + quadraticTerm(diffY(u, d), diffY(u, d),d)) 
end

# Parameters
parameters = Dict(
    "nu" => 0.5
)

t_span = [0, 2]

prob = SpectralODEProblem(L, N, domain, u0, t_span, p=parameters, dt=0.001)

## Solve and plot
sol = spectral_solve(prob, MSS3())

surface(domain, exp.(sol.u[end]), xlabel="x")

function HeatEquationAnalyticalSolution3(u0, domain, p, t)
    log.(exp.(-(domain.x'.^2 .+ domain.y.^2)/(2*(1 + 2*p["nu"]*t)))/(1 + 2*p["nu"]*t) .+ 1)
end

## Time convergence test
timesteps = [1e-1, 1e-2, 1e-3, 1e-4]
_, convergence1 = test_timestep_convergence(prob, HeatEquationAnalyticalSolution3, timesteps, MSS1())
_, convergence2 = test_timestep_convergence(prob, HeatEquationAnalyticalSolution3, timesteps, MSS2())
_, convergence3 = test_timestep_convergence(prob, HeatEquationAnalyticalSolution3, timesteps, MSS3())
plot(timesteps, convergence1, xaxis=:log, yaxis=:log, label="MSS1")
plot!(timesteps, convergence2, xaxis=:log, yaxis=:log, label="MSS2", color="dark green")
plot!(timesteps, convergence3, xaxis=:log, yaxis=:log, label="MSS3", color="orange")
plot!(timesteps, 0.5 * timesteps .^ 2, linestyle=:dash, label=L"\frac{1}{2}dt^2", xlabel="dt",
    ylabel=L"||U-u_a||", title="Timestep convergence, Nonlin-Diffusion (N =$(domain.Nx))", xticks=timesteps)
savefig("Timestep convergence, Nonlin-Diffusion (N =$(domain.Nx)).pdf")

## Resolution convergence test
resolutions = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] #[2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
_, convergence1 = test_resolution_convergence(prob, gaussian, HeatEquationAnalyticalSolution3, resolutions, MSS1())
_, convergence2 = test_resolution_convergence(prob, gaussian, HeatEquationAnalyticalSolution3, resolutions, MSS2())
_, convergence3 = test_resolution_convergence(prob, gaussian, HeatEquationAnalyticalSolution3, resolutions, MSS3())

plot(resolutions, convergence1, xaxis=:log2, yaxis=:log, label="MSS1")
plot!(resolutions, convergence2, xaxis=:log2, yaxis=:log, label="MSS2", color="dark green")
plot!(resolutions, convergence3, xaxis=:log2, yaxis=:log, label="MSS3", color="orange")
plot!(resolutions[1:end-4], 0.5 * exp.(-0.5 * resolutions)[1:end-4], label=L"\frac{1}{2}\exp\left(-\frac{N}{2}\right)", linestyle=:dash,
    xaxis=:log2, yaxis=:log, xticks=resolutions, xlabel=L"N_x \wedge N_y",
    ylabel=L"||U-u_a||/N_xN_y", title="Resolution convergence, Nonlin-Diffusion (dt=$(prob.dt))")
savefig("Resolution convergence, Nonlin-Diffusion (dt=$(prob.dt)).pdf")