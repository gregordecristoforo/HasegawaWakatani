## Run all (alt+enter)
include("../src/HasagawaWakatini.jl")

## Run scheme test for Burgers equation
#domain = Domain(512, 512, 200, 100, anti_aliased=false)
domain = Domain(256, 256, 50, 50, anti_aliased=false)
#domain = Domain(1024, 1024, 50, 50, anti_aliased=false)
u0 = gaussian.(domain.x', domain.y, A=1, B=0, l=1, x0 = -5) .+ gaussian.(domain.x', domain.y, A=-1, B=0, l=1, x0 = 5, y0=1)
contourf(u0)

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
t_span = [0, 25]

# The problem
prob = SpectralODEProblem(L, N, domain, [u0;;; zero(u0)], t_span, p=parameters, dt=1e-2)

# Array of diagnostics want
diagnostics = [
    ProbeDensityDiagnostic([(5, 0), (8.5, 0), (11.25, 0), (14.375, 0)], N=10),
    RadialCOMDiagnostic(),
    ProgressDiagnostic(100),
    CFLDiagnostic(),
    RadialCFLDiagnostic(100),
    PlotDensityDiagnostic(10)
]

# The output
output = Output(prob, 1, diagnostics) #progressDiagnostic

## Solve and plot
sol = spectral_solve(prob, MSS3(), output)

## Recreate Garcia et al. plots
display(heatmap(domain, sol.u[5][:, :, 1], levels=10, aspect_ratio=:equal))
display(heatmap(domain, sol.u[10][:, :, 1], levels=10, aspect_ratio=:equal))
display(heatmap(domain, sol.u[15][:, :, 1], levels=10, aspect_ratio=:equal))
display(heatmap(domain, sol.u[end][:, :, 1], levels=10, aspect_ratio=:equal))
display(heatmap(domain, sol.u[5][:, :, 2], levels=10, aspect_ratio=:equal, color=:jet))
display(heatmap(domain, sol.u[10][:, :, 2], levels=10, aspect_ratio=:equal, color=:jet))
display(heatmap(domain, sol.u[15][:, :, 2], levels=10, aspect_ratio=:equal, color=:jet))
display(heatmap(domain, sol.u[end][:, :, 2], levels=10, aspect_ratio=:equal, color=:jet))

## Save data
using JLD
save("reproduced data Garcia 2005 PoP.jld", "data", sol.u)

## Analyze data
data = zeros(length(output.diagnostics[1].data))
for i in eachindex(data)
    data[i] = sol.diagnostics[1].data[i][1]
end
plot(data)

data = Array(reshape(reduce(hcat, sol.diagnostics[1].data), size(sol.diagnostics[1].data[1])..., length(sol.diagnostics[1].data)))
data = Matrix{Number}(data')
plot(0:0.1:25-0.1, data)