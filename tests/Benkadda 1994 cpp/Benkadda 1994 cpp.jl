## Run all (alt+enter)
include(relpath(pwd(), @__DIR__) * "/src/HasegawaWakatini.jl")

## Run Benkadda simulations
domain = Domain(256, 256, 128, 128, anti_aliased=true)
# TODO find initial condition
ic = initial_condition_linear_stability(domain, 1e-3)

# Linear operator
function L(u, d, p, t)
    D_n = p["D"] .* diffusion(u, d)
    D_Ω = p["ν"] .* diffusion(u, d)
    [D_n;;; D_Ω]
end

# Non-linear operator, linearized
function N(u, d, p, t)
    n = @view u[:, :, 1]
    Ω = @view u[:, :, 2]
    ϕ = solvePhi(Ω, d)

    dn = -poissonBracket(ϕ, n, d)
    dn .-= diffY(ϕ, d)
    dn .-= p["σ"] * n

    dΩ = -poissonBracket(ϕ, Ω, d)
    dΩ .-= p["g"] * diffY(n, d)
    dΩ .+= p["σ"] * ϕ
    return [dn;;; dΩ]
end

# Parameters
parameters = Dict(
    "D" => 1e-2,
    "ν" => 1e-2,
    "g" => 1e-1,
    "σ" => 1e-3,
)

t_span = [0, 10000]

prob = SpectralODEProblem(L, N, domain, ic, t_span, p=parameters, dt=2e-3)

# Diagnostics
diagnostics = [
    ProgressDiagnostic(500),
    ProbeAllDiagnostic((0, 0), N=50),
    PlotDensityDiagnostic(1000),
    RadialFluxDiagnostic(600),
    KineticEnergyDiagnostic(1000),
    PotentialEnergyDiagnostic(1000),
    EnstropyEnergyDiagnostic(750),
    GetLogModeDiagnostic(50, :ky),
    CFLDiagnostic(50),
    RadialPotentialEnergySpectraDiagnostic(50),
    PoloidalPotentialEnergySpectraDiagnostic(50),
    RadialKineticEnergySpectraDiagnostic(50),
    PoloidalKineticEnergySpectraDiagnostic(50),
]

# Output
cd(relpath(@__DIR__, pwd()))
output = Output(prob, 1001, diagnostics, "output/benkadda april tewnthy fourth.h5", 
simulation_name=:parameters, store_locally=false)

FFTW.set_num_threads(16)

## Solve and plot
sol = spectral_solve(prob, MSS3(), output, resume=true)

# data = sol.simulation["fields"][:, :, :, :]
# t = sol.simulation["t"][:]
# default(legend=false)
# anim = @animate for i in axes(data, 4)
#     heatmap(data[:, :, 1, i], aspect_ratio=:equal, xaxis=L"x", yaxis=L"y", title=L"n(t=" * "$(round(t[i], digits=0)))")
# end
# gif(anim, "benkadda long.gif", fps=20)

send_mail("Long benkadda simulation finnished!")#, attachment="benkadda.gif")
close(output.file)

##--------------------------------- Data analysis ------------------------------------------
cd(relpath(@__DIR__, pwd()))
fid = h5open("output/benkadda april twelth long.h5", "r")
simulation = fid[keys(fid)[1]]

probe_data = read(simulation["Density probe/data"])
save("density probe benkadda.jld", "probe data", probe_data)
# t = read(simulation["Density probe/t"])

# plot(probe_data, marker=".")

# using Statistics
# using StatsPlots
# n = (probe_data .- mean(probe_data)) / std(probe_data)
# plot(n)

# density(n, minorticks=0.1, xlabel=L"(\tilde{n}-\langle \tilde{n}\rangle)/\tilde{n}_{rms}",
#     ylabel=L"P(n)", guidefontsize=13, titlefontsize=13, title="Histogram Benkadda", label="")

# Γ = -read(simulation["Radial flux/data"])[3000:end]
# plot(Γ, label="", xaxis=L"t", yaxis=L"Γ")

# Γ_n = (Γ .- mean(Γ)) / std(Γ)
# plot(Γ_n, label="", xaxis=L"t", yaxis=L"(\tilde{\Gamma}-\langle\tilde{\Gamma}\rangle)/\tilde{\Gamma}_rms",
#     minorticks=true, guidefontsize=13)

# P = read(simulation["Enstropy energy integral/data"])#[3000:end]
# plot(P[4000:15:5001], marker=".")