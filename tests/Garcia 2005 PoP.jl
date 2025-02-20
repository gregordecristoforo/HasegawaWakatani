## Run all (alt+enter)
include("../src/HasagawaWakatini.jl")

## Run scheme test for Burgers equation
#domain = Domain(1024, 1024, 50, 50, anti_aliased=false)
domain = Domain(256, 256, 50, 50, anti_aliased=false)
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
t_span = [0, 20]

# Speed up simulation
FFTW.set_num_threads(16)

# The problem
prob = SpectralODEProblem(L, N, domain, [u0;;; zero(u0)], t_span, p=parameters, dt=1e-2)

# Array of diagnostics want
diagnostics = [
    #ProbeDensityDiagnostic([(5, 0), (8.5, 0), (11.25, 0), (14.375, 0)], N=10),
    #RadialCOMDiagnostic(1),
    ProgressDiagnostic(100),
    CFLDiagnostic(1),
    #PlotDensityDiagnostic(1000),
    #PlotVorticityDiagnostic(1000),
    #PlotPotentialDiagnostic(1000),
]

# The output
output = Output(prob, 21, diagnostics)

## Solve and plot
@time sol = spectral_solve(prob, MSS3(), output)
# with views: 801.72 seconds
# without views:  772.099 seconds
a = 0
data = stack(diagnostics[2].data)

hj, kl = eachrow(data)

parentindices(kl)

# Folder path
cd("tests/Garcia 2005 Pop/")

## Recreate Garcia et al. plots Figure 1.
savefig(heatmap(domain, sol.u[6][:, :, 1], levels=10, aspect_ratio=:equal, xlabel=L"x", 
    ylabel=L"y", title=L"n(x, t = "*"$(round(sol.t[6], digits=2)))"), "blob density t=5.pdf")
savefig(heatmap(domain, sol.u[11][:, :, 1], levels=10, aspect_ratio=:equal, xlabel=L"x", 
    ylabel=L"y", title=L"n(x, t = "*"$(round(sol.t[11], digits=2)))"), "blob density t=10.pdf")
savefig(heatmap(domain, sol.u[16][:, :, 1], levels=10, aspect_ratio=:equal, xlabel=L"x", 
    ylabel=L"y", title=L"n(x, t = "*"$(round(sol.t[16], digits=2)))"), "blob density t=15.pdf")
savefig(heatmap(domain, sol.u[end][:, :, 1], levels=10, aspect_ratio=:equal, xlabel=L"x", 
    ylabel=L"y", title=L"n(x, t = "*"$(round(sol.t[end], digits=2)))"), "blob density t=20.pdf")
savefig(heatmap(domain, sol.u[6][:, :, 2], levels=10, aspect_ratio=:equal, xlabel=L"x", 
    ylabel=L"y", title=L"\Omega(x, t = "*"$(round(sol.t[6], digits=2)))"), "blob vorticity t=5.pdf")
savefig(heatmap(domain, sol.u[11][:, :, 2], levels=10, aspect_ratio=:equal, xlabel=L"x", 
    ylabel=L"y", title=L"\Omega(x, t = "*"$(round(sol.t[11], digits=2)))"), "blob vorticity t=10.pdf")
savefig(heatmap(domain, sol.u[16][:, :, 2], levels=10, aspect_ratio=:equal, xlabel=L"x", 
    ylabel=L"y", title=L"\Omega(x, t = "*"$(round(sol.t[16], digits=2)))"), "blob vorticity t=15.pdf")
savefig(heatmap(domain, sol.u[end][:, :, 2], levels=10, aspect_ratio=:equal, xlabel=L"x", 
    ylabel=L"y", title=L"\Omega(x, t = "*"$(round(sol.t[end], digits=2)))"), "blob vorticity t=20.pdf")

## Recreate Garcia et al. plots Figure 2.
plot(domain.x, vExB(sol.u[1], domain)[1][domain.Ny÷2, :]/2, linestyle=:dot, color=:blue, opacity=0.4)
plot!(domain.x, vExB(sol.u[6], domain)[1][domain.Ny÷2, :]/2, linestyle=:dot, color=:blue, opacity=0.4)
plot!(domain.x, vExB(sol.u[11], domain)[1][domain.Ny÷2, :]/2, linestyle=:dot, color=:blue, opacity=0.4)
plot!(domain.x, vExB(sol.u[16], domain)[1][domain.Ny÷2, :]/2, linestyle=:dot, color=:blue, opacity=0.4)
plot!(domain.x, vExB(sol.u[end], domain)[1][domain.Ny÷2, :]/2, linestyle=:dot, color=:blue, opacity=0.4)
plot!(domain.x, sol.u[1][domain.Ny÷2,:,1], color=:red)
plot!(domain.x, sol.u[6][domain.Ny÷2,:,1], color=:red)
plot!(domain.x, sol.u[11][domain.Ny÷2,:,1], color=:red)
plot!(domain.x, sol.u[16][domain.Ny÷2,:,1], color=:red)
plot!(domain.x, sol.u[end][domain.Ny÷2,:,1], color=:red, xlim=[-5,25], xlabel=L"x", 
ylabel=L"n(x,t), v_x(x,t)/2")
savefig("blob radial variations.pdf")

## Recreate Garcia et al. plots Figure 3.
probe_data = extract_diagnostic(sol.diagnostics[1].data)
t = sol.diagnostics[1].t
positions = sol.diagnostics[1].args 
plot(t, probe_data[1,:], xlabel=L"t", ylabel=L"u(x,t)", label=L"x="*"$(positions[1])")
plot!(t, probe_data[2,:], label=L"x="*"$(positions[2])")
plot!(t, probe_data[3,:], label=L"x="*"$(positions[3])")
plot!(t, probe_data[4,:], label=L"x="*"$(positions[4])")

## Recreate max velocity plot
tends = logspace(2, -2, 22)
dts = tends/2000
amplitudes = logspace(-2,5,22)
max_velocities = similar(amplitudes)
velocities = Vector{typeof(data)}(undef, length(amplitudes))

parameters = Dict(
    "nu" => 1e-3,
    "kappa" => 1e-3
)

for (i,A) in enumerate(amplitudes)
    # Update initial initial_condition
    u0 = gaussian.(domain.x', domain.y, A=A, B=0, l=1)
    # Update problem 
    prob = SpectralODEProblem(L, N, domain, [u0;;; zero(u0)], [0, tends[i]], p=parameters, dt=dts[i])
    # Reset diagnostics
    diagnostics = [RadialCOMDiagnostic(1), ProgressDiagnostic(100), PlotDensityDiagnostic(1000),]
    # Update output
    output = Output(prob, 21, diagnostics)
    # Solve 
    sol = spectral_solve(prob, MSS3(), output)
    # Extract velocity
    velocities[i] = extract_diagnostic(sol.diagnostics[1].data)
    # Determine max velocity
    max_velocities[i] = maximum(velocities[i][2,:])    
end

plot(amplitudes, max_velocities, xaxis=:log, yaxis=:log, marker=:circle)
plot(velocities[end][3,1:2000])
plot!(velocities[end][2,:])

for i in eachindex(velocities)
    max_velocities[i] = maximum(velocities[i][3,:])
end
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

## Debugging
sol.diagnostics[1].data
sol.diagnostics[3].data[1:400]
sol.diagnostics[3].t[200]
data = extract_diagnostic(sol.diagnostics[1].data)
plot(data[2,:])
plot(sol.diagnostics[1].t[1:30000], data[2,1:30000])

sol.diagnostics[1].t[argmax(data[2,:])]

0.02.*size(sol.diagnostics[1].data)
