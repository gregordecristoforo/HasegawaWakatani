## Run all (alt+enter)
include(relpath(pwd(), @__DIR__) * "/src/HasegawaWakatini.jl")

## Run Benkadda simulations
domain = Domain(256, 256, 128, 128, anti_aliased=true)
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
    dn .+= p["σ"] * ϕ

    dΩ = -poissonBracket(ϕ, Ω, d)
    dΩ .-= p["g"] * diffY(n, d)
    dΩ .+= p["σ"] * ϕ
    return [dn;;; dΩ]
end

# Parameters
parameters = Dict(
    "D" => 1e-2,
    "ν" => 1e-2, # nu
    "g" => 2e-2,
    "σ" => 1e-3,
)

t_span = [0, 10000]

prob = SpectralODEProblem(L, N, domain, ic, t_span, p=parameters, dt=1e-3)

# Diagnostics
diagnostics = [
    ProgressDiagnostic(1000),
    ProbeAllDiagnostic([(x,0) for x in LinRange(-40,50, 10)], N=100),
    #PlotDensityDiagnostic(1000),
    RadialFluxDiagnostic(600),
    KineticEnergyDiagnostic(1000),
    PotentialEnergyDiagnostic(1000),
    EnstropyEnergyDiagnostic(750),
    GetLogModeDiagnostic(50, :ky),
    CFLDiagnostic(500),
    RadialPotentialEnergySpectraDiagnostic(50),
    PoloidalPotentialEnergySpectraDiagnostic(50),
    RadialKineticEnergySpectraDiagnostic(50),
    PoloidalKineticEnergySpectraDiagnostic(50),
]

# Output
cd(relpath(@__DIR__, pwd()))
output = Output(prob, 1001, diagnostics, "../output/benkadda phi g=2e-2.h5",
    simulation_name=:parameters, store_locally=false)

FFTW.set_num_threads(16)

## Solve and plot
sol = spectral_solve(prob, MSS3(), output, resume=true)

send_mail("Benkadda (sigma phi) g=2e-2 simulation finnished!")
close(output.file)