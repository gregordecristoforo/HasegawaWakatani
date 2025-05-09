## Run all (alt+enter)
include(relpath(pwd(), @__DIR__) * "/src/HasegawaWakatini.jl")

## Run Benkadda simulations
domain = Domain(256, 256, 128, 128, anti_aliased=true)
# TODO find initial condition
ic = cu(initial_condition_linear_stability(domain, 1e-3))

# Linear operator
function L(u, d, p, t)
    D_n = p["D"] .* diffusion(u, d)
    D_Ω = p["ν"] .* diffusion(u, d)
    cat(D_n, D_Ω, dims=3) #[D_n;;; D_Ω]
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
    return cat(dn, dΩ, dims=3) #return [dn;;; dΩ]
end

# Parameters
parameters = Dict(
    "D" => 1e-2,
    "ν" => 1e-2,
    "g" => 1e-1,
    "σ" => 1e-3,
)

t_span = [0, 100]

prob = SpectralODEProblem(L, N, domain, ic, t_span, p=parameters, dt=1e-3)

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
output = Output(prob, 1001, [], "output/CUDA debug.h5", 
simulation_name=:parameters, store_locally=false, store_hdf=false)

FFTW.set_num_threads(16)

## Solve and plot
using BenchmarkTools
@time sol = spectral_solve(prob, MSS3(), output)
# 173 seconds

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

## Extract data from simulation batch to do local python analysis
fid = h5open("output/benkadda g=2e-2.h5", "r")

data = fid[keys(fid)[1]]["All probe/data"][:,:,:]
t = fid[keys(fid)[1]]["All probe/t"][:]

using JLD
jldopen("processed/all probes benkadda g=2e-2.jld", "w") do file
    g = create_group(file, "data")
    g["n"] = data[1,1,:]
    g["Omega"] = data[1,2,:]
    g["phi"] = data[1,3,:]
    g["vx"] = data[1,4,:]
    g["Gamma"] = data[1,5,:]
    g["t"] = t
end







































domain = Domain(256, 256, 128, 128, anti_aliased=true)

domain.kx
domain.ky
domain.x 
domain.y

domain.SC.Laplacian
domain.SC.invLaplacian
domain.SC.HyperLaplacian
domain.SC.DiffX
domain.SC.DiffY
domain.SC.DiffXX
domain.SC.DiffYY

FT = domain.transform.FT
u = ic[:,:,1]
u_hat = FT*u

diffX(u_hat, domain)           # ✓
diffY(u_hat, domain)           # ✓
diffXX(u_hat, domain)          # ✓
diffYY(u_hat, domain)          # ✓
diffusion(u_hat, domain)       # ✓
hyper_diffusion(u_hat, domain) # ✓
solvePhi(u_hat, domain)        # ✓

spectral_transform!(u_hat, u, domain.transform.FT) # ✓

# QuadraticTerms cache
domain.SC.up                  # ✓
domain.SC.vp                  # ✓
domain.SC.U                   # ✓
domain.SC.V                   # ✓
domain.SC.qtl                 # ✓
domain.SC.qtr                 # ✓
domain.SC.phi                 # ✓
domain.SC.QTPlans.FT          # ✓
domain.SC.QTPlans.iFT         # ✓

# More complex methods
spectral_exp(u_hat, domain)   # ✓
spectral_expm1(u_hat, domain) # ✓
spectral_log(u_hat, domain)   # ✓

# Methods utilizing threading
quadraticTerm(diffXX(u_hat, domain), diffYY(u_hat, domain), domain) # ✓
poissonBracket(u_hat, u_hat, domain)                                # ✓

p = Dict(
    "D" => Float32(1e-2),
    "ν" => Float32(1e-2),
    "g" => Float32(1e-1),
    "σ" => Float32(1e-3),
)

L(u_hat, domain, p , 0)

uu_hat = cat(u_hat, u_hat, dims=3)
N(uu_hat, domain, p , 0)

prob = SpectralODEProblem(L, N, domain, ic, t_span, p=p, dt=2e-3)
cache = get_cache(prob, MSS3())
cache.c

if isnan.(u_hat)
    println("hi")
end