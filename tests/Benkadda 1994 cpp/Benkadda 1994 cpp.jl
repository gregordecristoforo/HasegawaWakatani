## Run all (alt+enter)
include(relpath(pwd(), @__DIR__) * "/src/HasegawaWakatini.jl")

## Run Benkadda simulations
domain = Domain(256, 256, 128, 128, use_cuda=true)

# TODO find initial condition
ic = initial_condition_linear_stability(domain, 1e-3)

# Linear operator
function L(u, d, p, t)
    D_n = p["D"] .* diffusion(u, d)
    D_Ω = p["ν"] .* diffusion(u, d)
    cat(D_n, D_Ω, dims=3) #[D_n;;; D_Ω]
end

# Non-linear operator, τ=0
function N(u, d, p, t)
    n = @view u[:, :, 1]
    Ω = @view u[:, :, 2]
    ϕ = solvePhi(Ω, d)

    dn = -poissonBracket(ϕ, n, d)
    dn .-= diffY(ϕ, d)
    dn .+= p["σ"] * ϕ

    dΩ = -poissonBracket(ϕ, Ω, d)
    dΩ .-= p["g"] * diffY(n, d)
    dΩ .+= p["σ"] * ϕ
    return cat(dn, dΩ, dims=3) #return [dn;;; dΩ]
end

# Non-linear operator, τ≠0
# function N(u, d, p, t)
#     n = @view u[:, :, 1]
#     Ω = @view u[:, :, 2]
#     ϕ = solvePhi(Ω, d)

#     dn = -poissonBracket(ϕ, n, d)
#     dn .-= diffY(ϕ, d)
#     dn .+= p["σ"] * ϕ

#     dΩ = -poissonBracket(ϕ, Ω, d)
#     dΩ .-= poissonBracket(p["τ"] * n, Ω, d)
#     dΩ .+= p["τ"] *poissonBracket(diffX(ϕ,d), diffX(n,d), d)
#     dΩ .+= p["τ"] *poissonBracket(diffY(ϕ,d), diffY(n,d), d)
#     dΩ .-= p["g"] * diffY(n, d)
#     dΩ .+= p["σ"] * ϕ
#     dΩ .+= p["τ"] * diffY(Ω, d)
#     return cat(dn, dΩ, dims=3) #return [dn;;; dΩ]
# end

# Parameters
parameters = Dict(
    "D" => 1e-2,
    "ν" => 1e-2,
    "g" => 1e-1,
    "σ" => 1e-1,
    #"τ" => 1,
)

t_span = [0, 300]

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
output = Output(prob, 1001, diagnostics, "output/CUDA debug.h5",
    simulation_name="gpu6", store_locally=false, store_hdf=true);

FFTW.set_num_threads(16)

## Solve and plot
#using BenchmarkTools
sol = spectral_solve(prob, MSS3(), output, resume=false)

# # Difference at ~800
# data_gpu = sol.file["gpu/fields"][:,:,1,580]
# data_cpu = sol.file["gpu3/fields"][:,:,1,900]
# heatmap(data_cpu)
# heatmap(data_gpu)
# heatmap(data_gpu-data_cpu)

data = sol.file["gpu6/fields"][:,:,1,700]

contour(transpose(data), levels=17,cbar=false, clabels=false, color=:black)

# CUDA.pool_status()

# ic = copy(output.file["cpu/fields"][:,:,:,1])
# # data = sol.simulation["fields"][:, :, :, :]
# # t = sol.simulation["t"][:]
# # default(legend=false)
# # anim = @animate for i in axes(data, 4)
# #     heatmap(data[:, :, 1, i], aspect_ratio=:equal, xaxis=L"x", yaxis=L"y", title=L"n(t=" * "$(round(t[i], digits=0)))")
# # end
# # gif(anim, "benkadda long.gif", fps=20)

# send_mail("Long benkadda simulation finnished!")#, attachment="benkadda.gif")
# close(output.file)

# ##--------------------------------- Data analysis ------------------------------------------
# cd(relpath(@__DIR__, pwd()))
# fid = h5open("output/benkadda april twelth long.h5", "r")
# simulation = fid[keys(fid)[1]]

# probe_data = read(simulation["Density probe/data"])
# save("density probe benkadda.jld", "probe data", probe_data)
# # t = read(simulation["Density probe/t"])

# # plot(probe_data, marker=".")

# # using Statistics
# # using StatsPlots
# # n = (probe_data .- mean(probe_data)) / std(probe_data)
# # plot(n)

# # density(n, minorticks=0.1, xlabel=L"(\tilde{n}-\langle \tilde{n}\rangle)/\tilde{n}_{rms}",
# #     ylabel=L"P(n)", guidefontsize=13, titlefontsize=13, title="Histogram Benkadda", label="")

# # Γ = -read(simulation["Radial flux/data"])[3000:end]
# # plot(Γ, label="", xaxis=L"t", yaxis=L"Γ")

# # Γ_n = (Γ .- mean(Γ)) / std(Γ)
# # plot(Γ_n, label="", xaxis=L"t", yaxis=L"(\tilde{\Gamma}-\langle\tilde{\Gamma}\rangle)/\tilde{\Gamma}_rms",
# #     minorticks=true, guidefontsize=13)

# # P = read(simulation["Enstropy energy integral/data"])#[3000:end]
# # plot(P[4000:15:5001], marker=".")

# ## Extract data from simulation batch to do local python analysis
# fid = h5open("output/benkadda g=2e-2.h5", "r")

# data = fid[keys(fid)[1]]["All probe/data"][:,:,:]
# t = fid[keys(fid)[1]]["All probe/t"][:]

# using JLD
# jldopen("processed/all probes benkadda g=2e-2.jld", "w") do file
#     g = create_group(file, "data")
#     g["n"] = data[1,1,:]
#     g["Omega"] = data[1,2,:]
#     g["phi"] = data[1,3,:]
#     g["vx"] = data[1,4,:]
#     g["Gamma"] = data[1,5,:]
#     g["t"] = t
# end









# #domain.





# # poissonBrackets differ!
# maximum(real(Array(pb1)-pb2))
# maximum(real(Array(pbb1)-pbb2))


# e1 = spectral_exp(ϕ1, domain)
# e2 = spectral_exp(ϕ2, domain2)

# maximum(real(Array(e1)-e2))



# domain = Domain(256, 256, 128, 128, use_cuda=true,anti_aliased=true)
# domain2 = Domain(256, 256, 128, 128, use_cuda=false,anti_aliased=true)

# n = CuArray(ic[:,:,1])
# n2 = copy(ic[:,:,1])
# o = CuArray(ic[:,:,2])
# o2 = copy(ic[:,:,2])


# n_hat = domain.transform.FT*n
# n2_hat = domain2.transform.FT*n2

# pb = poissonBracket(n_hat, O_hat, domain)
# pbb = poissonBracket(n2_hat, O2_hat, domain2)

# heatmap(abs.(transpose(Array(n_hat))-n2_hat),)


# n = CuArray(n)

# n_hat = rfft(n)
# n2_hat = rfft(n2)
# O_hat = rfft(o)
# O2_hat = rfft(o2)


# r = spectral_conv!(domain.SC.qtl, n_hat, n_hat, domain.SC)
# r2 = spectral_conv!(domain2.SC.qtl, n2_hat, n2_hat, domain2.SC)
# # Compare with a tolerance
# @test isapprox(Array(pb), pbb; rtol=1e-16, atol=1e-16)

# heatmap(abs.(Array(r)-r2))

# SC = domain.SC
# plans = SC.QTPlans
# mul!(SC.U, plans.iFT, SC.padded ? pad!(SC.up, n_hat, plans) : n_hat)
# mul!(SC.V, plans.iFT, SC.padded ? pad!(SC.vp, n_hat, plans) : n_hat)
# @. SC.U *= SC.V
# mul!(SC.padded ? SC.up : SC.qtl, plans.FT, SC.U)

# SC2 = domain2.SC
# plans2 = SC2.QTPlans
# mul!(SC2.U, plans2.iFT, SC2.padded ? pad!(SC2.up, n2_hat, plans2) : copy(n2_hat))
# mul!(SC2.V, plans2.iFT, SC2.padded ? pad!(SC2.vp, n2_hat, plans2) : copy(n2_hat))
# @. SC2.U *= SC2.V
# mul!(SC2.padded ? SC2.up : SC2.qtl, plans2.FT, SC2.U)

# heatmap(SC2.U)
# heatmap(Array(SC.V))
# heatmap(Array(n2))


# heatmap(domain2.transform.iFT*Array(n2_hat))


# b = copy(SC.U)

# @. SC.U *= SC.V

# maximum(SC.U - b*SC.V)