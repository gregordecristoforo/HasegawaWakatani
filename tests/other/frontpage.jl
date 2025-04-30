## Run all (alt+enter)
include(relpath(pwd(), @__DIR__) * "/src/HasegawaWakatini.jl")

# High resolution domain for frontpage matter
domain = Domain(1024, 1024, 100, 100, anti_aliased=true)
ic = initial_condition_linear_stability(domain, 1e-3)

# Linear operator
function L(u, d, p, t)
    D_n = p["D_n"] .* diffusion(u, d)
    D_Ω = p["D_Ω"] .* diffusion(u, d)
    [D_n;;; D_Ω]
end

# Non-linear operator, linearized
function N(u, d, p, t)
    n = @view u[:, :, 1]
    Ω = @view u[:, :, 2]
    ϕ = solvePhi(Ω, d)

    dn = -poissonBracket(ϕ, n, d)
    dn .-= (p["kappa"] - p["g"]) * diffY(ϕ, d)
    dn .-= p["g"] * diffY(n, d)
    dn .-= p["sigma_n"] * n

    dΩ = -poissonBracket(ϕ, Ω, d)
    dΩ .-= p["g"] * diffY(n, d)
    dΩ .+= p["sigma_Ω"] * ϕ
    return [dn;;; dΩ]
end

# Parameters
parameters = Dict(
    "D_Ω" => 1e-2,
    "D_n" => 1e-2,
    "g" => 2e-3,
    "sigma_Ω" => 1e-3,
    "sigma_n" => 1e-3,
    "kappa" => sqrt(1e-1)
)

t_span = [0, 5_000]

prob = SpectralODEProblem(L, N, domain, ic, t_span, p=parameters, dt=1e-2)

# Diagnostics
diagnostics = [PlotDensityDiagnostic(5000), ProgressDiagnostic(100)]

# Output
cd(relpath(@__DIR__, pwd()))
output = Output(prob, 1001, diagnostics, "frontpage matter.h5", simulation_name="2025-04-26T15:26:28.052",
store_locally = false)

FFTW.set_num_threads(16)

# Solve and plot
sol = spectral_solve(prob, MSS3(), output, resume=true)

##

# data = sol.simulation["fields"][:, :, :, :]
# t = sol.simulation["t"][:]
# default(legend=false)
# anim = @animate for i in axes(data, 4)
#     heatmap(data[:, :, 1, i], aspect_ratio=:equal, xaxis=L"x", yaxis=L"y", title=L"n(t=" * "$(round(t[i], digits=0)))")
# end
# gif(anim, "long timeseries.gif", fps=20)

colors = ["#003f5d", "#125770", "#b3c4cc", "#0e3b51", "#01223d"]
colors = ["#003f5d", "#125770", "#01223d", "#0e3b51", "#b3c4cc"]
colors = ["#00203b", "#0e3b52" ,"#0e556e", "#b3c4cc","#003e5d"]

cgrad(colors)

u0_hat = sol.simulation["cache_backup/u"][:, :, :]
n = domain.transform.iFT * u0_hat[:, :, 1]

n = sol.simulation["fields"][:,:,1,231]

heatmap(sol.simulation["fields"][:,:,1,239], colormap=cgrad(colors), axis=nothing, aspect_ratio=:equal,
    framestyle=:none, colorbar=false, size=(1024 / 3, 1024 / 3), widen=false, dpi=300,
    margin=-10Plots.mm)
savefig("frontpage.pdf")

send_mail("g=2e-3 finnished, go analyse the data!")
close(output.file)

#sol.diagnostics[3].data[501]
# n = data[1,1,:]
# Ω = data[1,2,:]
# ϕ = data[1,3,:]
# v_x = data[1,4,:]
# Γ = data[1,5,:]
