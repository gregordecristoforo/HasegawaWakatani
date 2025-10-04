## Run all (alt+enter)
using HasegawaWakatani
using CUDA
using Plots

## Run scheme test for Burgers equation
domain = Domain(128, 128, Lx=160, Ly=160)
ic = initial_condition(random_crossphased, domain, value=1e-3)
ic[:, :, 1] .+= 0.5

# Linear operator
function Linear(du, u, d, p, t)
    @unpack ν, μ = p
    n, Ω = eachslice(u, dims=3)
    dn, dΩ = eachslice(du, dims=3)

    dn .= ν .* laplacian(n, d)
    dΩ = μ .* laplacian(Ω, d)
end

# Current implementation of source
function source(x, y, S_0, λ_s)
    @. S_0 * exp(-(x / λ_s)^2) + 0 * y
end

S = get_fwd(domain) * CuArray(source(domain.x', domain.y, 5e-4, 5))
heatmap(Array(get_bwd(domain) * S), aspect_ratio=:equal)

# Non-linear operator, fully non-linear
function NonLinear(du, u, d, p, t)
    @unpack g, σ_0 = p
    n, Ω = eachslice(u, dims=3)
    dn, dΩ = eachslice(du, dims=3)
    ϕ = solve_phi(Ω, d)

    dn .= poisson_bracket(n, ϕ, d) .- g * diff_y(n, d) .+ g * quadratic_term(n, diff_y(ϕ, d), d) .- σ_0 * quadratic_term(n, spectral_exp(-ϕ, d), d) .+ S
    dΩ .= poisson_bracket(Ω, ϕ, d) .- g * diff_y(spectral_log(n, d), d) .- σ_0 * spectral_expm1(-ϕ, d)
end

# Parameters
parameters = (μ=1e-2, ν=1e-2, g=8e-4, σ_0=2e-4, S_0=5e-4, λ_s=5.0)

t_span = [0, 50_000_000]

prob = SpectralODEProblem(Linear, NonLinear, ic, domain, t_span, p=parameters, dt=1)

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
output = Output(prob, filename="Bisai.h5", diagnostics=diagnostics, stride=100,
    simulation_name=:parameters, store_locally=false)

# Solve and plot
sol = spectral_solve(prob, MSS3(), output, resume=true)

# bisai_php_19_052509 gives Lₓ = L_y = 160ρₛ, g = 1e-3, 128-modes
# bisai_php_12_072520 gives dt = \omega_c^{-1}, g = 8e-4