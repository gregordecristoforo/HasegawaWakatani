## Run all (alt+enter)
include(relpath(pwd(), @__DIR__) * "/src/HasegawaWakatini.jl")

## Run scheme test for Burgers equation
domain = Domain(128, 128, 160, 160, anti_aliased=true)
ic = initial_condition_linear_stability(domain, 1e-3)
ic[:, :, 1] .+= 0.5

heatmap(ic[:, :, 1])

# Linear operator
function L(u, d, p, t)
    D_n = p["D_n"] .* diffusion(u, d)
    D_Ω = p["D_Ω"] .* diffusion(u, d)
    cat(D_n, D_Ω, dims=3)
end

function source(x, y, S_0, λ_s)
    @. S_0 * exp(-(x / λ_s)^2) + 0 * y
end

S = domain.transform.FT * CuArray(source(domain.x', domain.y, 5e-4, 5))

# Non-linear operator, fully non-linear
function N(u, d, p, t)
    n = @view u[:, :, 1]
    Ω = @view u[:, :, 2]
    ϕ = solve_phi(Ω, d)

    dn = -poisson_bracket(ϕ, n, d)
    dn .-= p["g"] * diff_y(n, d)
    dn += p["g"] * quadratic_term(n, diff_y(ϕ, d), d)
    dn .-= p["σ_0"] * quadratic_term(n, spectral_exp(-ϕ, d), d)
    # Plus the const source
    dn .+= S

    dΩ = -poisson_bracket(ϕ, Ω, d)
    dΩ .-= p["g"] * diff_y(spectral_log(n, d), d)
    dΩ .-= p["σ_0"] * spectral_expm1(-ϕ, d)
    return cat(dn, dΩ, dims=3)
end

# Parameters
parameters = Dict(
    "D_Ω" => 1e-2,
    "D_n" => 1e-2,
    "g" => 8e-4,
    "σ_0" => 2e-4,
    "S_0" => 5e-4,
    "λ_s" => 5.0,
)

t_span = [0, 50_000_000]

prob = SpectralODEProblem(L, N, domain, ic, t_span, p=parameters, dt=1)

# Diagnostics
diagnostics = [
    ProgressDiagnostic(1000),
    ProbeAllDiagnostic([(x, 0) for x in LinRange(-80, 64, 10)], N=10),
    PlotDensityDiagnostic(50),
    RadialFluxDiagnostic(50),
    KineticEnergyDiagnostic(50),
    PotentialEnergyDiagnostic(50),
    EnstropyEnergyDiagnostic(50),
    GetLogModeDiagnostic(50, :ky),
    CFLDiagnostic(50),
    RadialPotentialEnergySpectraDiagnostic(50),
    PoloidalPotentialEnergySpectraDiagnostic(50),
    RadialKineticEnergySpectraDiagnostic(50),
    PoloidalKineticEnergySpectraDiagnostic(50),
]

# Output
cd(relpath(@__DIR__, pwd()))
output = Output(prob, 1001, diagnostics, "output/Bisai.h5",
    simulation_name="last run", store_locally=false)

FFTW.set_num_threads(16)

## Solve and plot
sol = spectral_solve(prob, MSS3(), output, resume=true)

# data = sol.simulation["fields"][:, :, :, :]
# t = sol.simulation["t"][:]
# default(legend=false)
# anim = @animate for i in axes(data, 4)
#     heatmap(data[:, :, 1, i], aspect_ratio=:equal, xaxis=L"x", yaxis=L"y", title=L"n(t=" * "$(round(t[i], digits=0)))")
# end
# gif(anim, "long timeseries.gif", fps=20)

send_mail("Bisai simulation finnished!")
close(output)

# bisai_php_19_052509 gives Lₓ = L_y = 160ρₛ, g = 1e-3, 128-modes
# bisai_php_12_072520 gives dt = \omega_c^{-1}, g = 8e-4

fid = h5open("output/Bisai.h5")
sim = fid["last run"]

heatmap(sim["fields"][:, :, 1, 1000])
savefig("Bisai example.pdf")