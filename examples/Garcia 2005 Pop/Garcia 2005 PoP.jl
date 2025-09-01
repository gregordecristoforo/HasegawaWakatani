## Run all (alt+enter)
using HasegawaWakatani
using LaTeXStrings
using Plots

## Run scheme test for Burgers equation
domain = Domain(1024, 1024, 50, 50, anti_aliased=false)
#domain = Domain(256, 256, 50, 50, anti_aliased=false)
u0 = gaussian.(domain.x', domain.y, A=1, B=0, l=1)

# Linear operator
function L(u, d, p, t)
    D_θ = p["kappa"] * diffusion(u, d)
    D_Ω = p["nu"] * diffusion(u, d)
    [D_θ;;; D_Ω]
end

# Non-linear operator
function N(u, d, p, t)
    θ = @view u[:, :, 1]
    Ω = @view u[:, :, 2]
    ϕ = solvePhi(Ω, d)
    dθ = -poissonBracket(ϕ, θ, d)
    dΩ = -poissonBracket(ϕ, Ω, d)
    dΩ .-= diffY(θ, d)
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
#FFTW.set_num_threads(16)

# The problem
prob = SpectralODEProblem(L, N, domain, [u0;;; zero(u0)], t_span, p=parameters, dt=1e-3)

# Array of diagnostics want
diagnostics = [
    #ProbeDensityDiagnostic([(5, 0), (8.5, 0), (11.25, 0), (14.375, 0)], N=10),
    #RadialCOMDiagnostic(1),
    PlotDensityDiagnostic(5),
    ProgressDiagnostic(100),
    CFLDiagnostic(1),
    #PlotDensityDiagnostic(1000),
    #PlotVorticityDiagnostic(1000),
    #PlotPotentialDiagnostic(1000),
]

# Folder path
cd(relpath(@__DIR__, pwd()))

# The output
output = Output(prob, 21, diagnostics, "Garcia 2005 PoP.h5")

## Solve and plot
sol = spectral_solve(prob, MSS3(), output)

## Recreate Garcia et al. plots Figure 1.
savefig(heatmap(domain, sol.u[6][:, :, 1], levels=10, aspect_ratio=:equal, xlabel=L"x", ylim=[-10, 10], size=(600, 280),
        ylabel=L"y", title=L"n(x, t = " * "$(round(sol.t[6], digits=2)))"), "blob density t=5.pdf")
savefig(heatmap(domain, sol.u[11][:, :, 1], levels=10, aspect_ratio=:equal, xlabel=L"x", ylim=[-10, 10], size=(600, 280),
        ylabel=L"y", title=L"n(x, t = " * "$(round(sol.t[11], digits=2)))"), "blob density t=10.pdf")
savefig(heatmap(domain, sol.u[16][:, :, 1], levels=10, aspect_ratio=:equal, xlabel=L"x", ylim=[-10, 10], size=(600, 280),
        ylabel=L"y", title=L"n(x, t = " * "$(round(sol.t[16], digits=2)))"), "blob density t=15.pdf")
savefig(heatmap(domain, sol.u[end][:, :, 1], levels=10, aspect_ratio=:equal, xlabel=L"x", ylim=[-10, 10], size=(600, 280),
        ylabel=L"y", title=L"n(x, t = " * "$(round(sol.t[end], digits=2)))"), "blob density t=20.pdf")
savefig(heatmap(domain, sol.u[6][:, :, 2], levels=10, aspect_ratio=:equal, xlabel=L"x", ylim=[-10, 10], size=(600, 280),
        ylabel=L"y", title=L"\Omega(x, t = " * "$(round(sol.t[6], digits=2)))", color=:jet), "blob vorticity t=5.pdf")
savefig(heatmap(domain, sol.u[11][:, :, 2], levels=10, aspect_ratio=:equal, xlabel=L"x", ylim=[-10, 10], size=(600, 280),
        ylabel=L"y", title=L"\Omega(x, t = " * "$(round(sol.t[11], digits=2)))", color=:jet), "blob vorticity t=10.pdf")
savefig(heatmap(domain, sol.u[16][:, :, 2], levels=10, aspect_ratio=:equal, xlabel=L"x", ylim=[-10, 10], size=(600, 280),
        ylabel=L"y", title=L"\Omega(x, t = " * "$(round(sol.t[16], digits=2)))", color=:jet), "blob vorticity t=15.pdf")
savefig(heatmap(domain, sol.u[end][:, :, 2], levels=10, aspect_ratio=:equal, xlabel=L"x", ylim=[-10, 10], size=(600, 280),
        ylabel=L"y", title=L"\Omega(x, t = " * "$(round(sol.t[end], digits=2)))", color=:jet), "blob vorticity t=20.pdf")

plot(layout=grid(4, 2), legend=:none, size=(380, 400), dpi=400)
for i in 1:4
    heatmap!(domain.x, domain.y, sol.u[1+5i][:, :, 1], ylim=[-10, 10], aspect_ratio=:equal, subplot=2i - 1)
    heatmap!(domain.x, domain.y, sol.u[1+5i][:, :, 2], ylim=[-10, 10], aspect_ratio=:equal, subplot=2i, color=:jet)
end
savefig("blob evolution.png")

## Recreate Garcia et al. plots Figure 2.
plot(domain.x, sol.u[end][domain.Ny÷2, :, 1], color=:red, xlim=[-5, 25], xlabel=L"x",
    ylabel=L"n(x,t), v_x(x,t)/2", label=L"n(x,t)")
plot!(domain.x, vExB(sol.u[1], domain)[1][domain.Ny÷2, :] / 2, linestyle=:dot, color=:blue, opacity=0.4, label="")
plot!(domain.x, vExB(sol.u[6], domain)[1][domain.Ny÷2, :] / 2, linestyle=:dot, color=:blue, opacity=0.4, label="")
plot!(domain.x, vExB(sol.u[11], domain)[1][domain.Ny÷2, :] / 2, linestyle=:dot, color=:blue, opacity=0.4, label="")
plot!(domain.x, vExB(sol.u[16], domain)[1][domain.Ny÷2, :] / 2, linestyle=:dot, color=:blue, opacity=0.4, label="")
plot!(domain.x, vExB(sol.u[end], domain)[1][domain.Ny÷2, :] / 2, linestyle=:dot, color=:blue, opacity=0.4, label=L"v_x(x,t)/2")
plot!(domain.x, sol.u[1][domain.Ny÷2, :, 1], color=:red, label="")
plot!(domain.x, sol.u[6][domain.Ny÷2, :, 1], color=:red, label="")
plot!(domain.x, sol.u[11][domain.Ny÷2, :, 1], color=:red, label="")
plot!(domain.x, sol.u[16][domain.Ny÷2, :, 1], color=:red, label="")
plot!(domain.x, sol.u[end][domain.Ny÷2, :, 1], color=:red, xlim=[-5, 25], xlabel=L"x",
    ylabel=L"n(x,t), v_x(x,t)/2", label="")
savefig("blob radial variations.pdf")

## Recreate Garcia et al. plots Figure 3.
probe_data = extract_diagnostic(sol.diagnostics[1].data)
t = sol.diagnostics[1].t
positions = sol.diagnostics[1].args[1]
plot(t, probe_data[1, :], xlabel=L"t", ylabel=L"u(x,t)", label=L"x=" * "$(positions[1])")
plot!(t, probe_data[2, :], label=L"x=" * "$(positions[2])")
plot!(t, probe_data[3, :], label=L"x=" * "$(positions[3])")
plot!(t, probe_data[4, :], label=L"x=" * "$(positions[4])")
savefig("blob probe data.pdf")

## Recreate max velocity plot
tends = logspace(2, -2, 22)
dts = tends / 2000
amplitudes = logspace(-2, 5, 22)
max_velocities = similar(amplitudes)
velocities = Vector{Matrix}(undef, length(amplitudes))

parameters = Dict(
    "nu" => 1e-3,
    "kappa" => 1e-3
)

for (i, A) in enumerate(amplitudes)
    # Update initial initial_condition
    u0 = gaussian.(domain.x', domain.y, A=A, B=0, l=1)
    # Update problem 
    prob = SpectralODEProblem(L, N, domain, [u0;;; zero(u0)], [0, tends[i]], p=parameters, dt=dts[i])
    # Reset diagnostics
    diagnostics = [RadialCOMDiagnostic(1), ProgressDiagnostic(100), PlotDensityDiagnostic(1000),]
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
    max_velocities[i] = maximum(velocities[i][2, :])
end

plot(amplitudes, max_velocities, xaxis=:log, yaxis=:log, marker=:circle, xlabel=L"\Delta n/N", ylabel="max " * L"V", label="")
savefig("blob velocity linear.pdf")