## Run all (alt+enter)
include("../src/HasagawaWakatini.jl")

## Run scheme test for Burgers equation
#domain = Domain(512, 512, 200, 100, anti_aliased=false)
domain = Domain(1024, 1024, 50, 50, anti_aliased=false)
#domain = Domain(1024, 1024, 50, 50, anti_aliased=false)
u0 = gaussian.(domain.x', domain.y, A=1, B=0, l=1)

# Linear operator
function L(u, d, p, t)
    D_θ = p["kappa"] * diffusion(u, d)
    D_Ω = p["nu"] * diffusion(u, d)
    [D_θ;;; D_Ω]
end

# Non-linear operator
function N(u, d, p, t)
    θ = u[:, :, 1]
    Ω = u[:, :, 2]
    ϕ = solvePhi(Ω, d)
    dθ = -poissonBracket(ϕ, θ, d)
    dΩ = -poissonBracket(ϕ, Ω, d)
    dΩ -= diffY(θ, d)
    return [dθ;;; dΩ]
end

# Parameters
parameters = Dict(
    "nu" => 1e-2,
    "kappa" => 1e-2
)

# Time interval
t_span = [0, 50]

# The problem
prob = SpectralODEProblem(L, N, domain, [u0;;; zero(u0)], t_span, p=parameters, dt=1e-3)

# Array of diagnostics want
diagnostics = [
    #ProbeDensityDiagnostic([(5, 0), (8.5, 0), (11.25, 0), (14.375, 0)], N=10),
    RadialCOMDiagnostic(1),
    ProgressDiagnostic(10),
    #CFLDiagnostic(),
    #RadialCFLDiagnostic(100),
    PlotDensityDiagnostic(1000),
    PlotVorticityDiagnostic(1000),
    PlotPotentialDiagnostic(1000),
]

# The output
output = Output(prob, 21, diagnostics)

## Solve and plot
sol = spectral_solve(prob, MSS3(), output)

data = extract_diagnostic(sol.diagnostics[1].data)
plot(data[1,:])
plot(data[2,:])

## Recreate Garcia et al. plots
display(heatmap(sol.u[6][:, :, 1], levels=10, aspect_ratio=:equal))
display(heatmap(sol.u[11][:, :, 1], levels=10, aspect_ratio=:equal))
display(heatmap(sol.u[16][:, :, 1], levels=10, aspect_ratio=:equal))
display(heatmap(sol.u[end][:, :, 1], levels=10, aspect_ratio=:equal))
display(heatmap(sol.u[6][:, :, 2], levels=10, aspect_ratio=:equal, color=:jet))
display(heatmap(sol.u[11][:, :, 2], levels=10, aspect_ratio=:equal, color=:jet))
display(heatmap(sol.u[16][:, :, 2], levels=10, aspect_ratio=:equal, color=:jet))
display(heatmap(sol.u[end][:, :, 2], levels=10, aspect_ratio=:equal, color=:jet))

## Recreate max velocity plot
amplitudes = logspace(-2,5,22)
max_velocities = similar(amplitudes)
velocities = Vector{typeof(data)}(undef, length(amplitudes))

for (i,A) in enumerate(amplitudes)
    # Update initial initial_condition
    u0 = gaussian.(domain.x', domain.y, A=A, B=0, l=1)
    # Update problem 
    prob = SpectralODEProblem(L, N, domain, [u0;;; zero(u0)], t_span, p=parameters, dt=1e-2)
    # Update output
    output = Output(prob, 21, diagnostics)
    # Solve 
    sol = spectral_solve(prob, MSS3(), output)
    # Extract velocity
    velocities[i] = extract_diagnostic(sol.diagnostics[1].data)
    # Determine max velocity
    max_velocities[i] = maximum(velocities[i][2,:])    
end

plot(amplitudes, max_velocities, xaxis=:log)
save("velocitydata.jld", "data", velocities)

## Save data
using JLD
save("reproduced data Garcia 2005 PoP.jld", "data", sol.u)

## Analyze data
data = zeros(length(output.diagnostics[1].data))
for i in eachindex(data)
    data[i] = sol.diagnostics[1].data[i][4]
end
plot!(data)

data = Array(reshape(reduce(hcat, sol.diagnostics[1].data), size(sol.diagnostics[1].data[1])..., length(sol.diagnostics[1].data)))
data = Matrix{Number}(data')
plot(0:0.1:25-0.1, data)