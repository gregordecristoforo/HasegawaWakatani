## Run all (alt+enter)
include(relpath(pwd(), @__DIR__) * "/src/HasegawaWakatini.jl")

## Run scheme test for Burgers equation
domain = Domain(128, 128, 100, 100, anti_aliased=true)
ic = initial_condition_linear_stability(domain, 1e-6)

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

t_span = [0, 5_000_000]

prob = SpectralODEProblem(L, N, domain, ic, t_span, p=parameters, dt=1e-1)

# Diagnostics
diagnostics = [
    ProgressDiagnostic(1000),
    ProbeAllDiagnostic((0, 0), N=100),
    #PlotDensityDiagnostic(5000),
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
output = Output(prob, 1001, diagnostics, "../output/sheath-interchange g=2e-3.h5",
simulation_name=:parameters)
    
FFTW.set_num_threads(16)

## Solve and plot
sol = spectral_solve(prob, MSS3(), output)

send_mail("g=2e-3 finnished, go analyse the data!")
close(output.file)