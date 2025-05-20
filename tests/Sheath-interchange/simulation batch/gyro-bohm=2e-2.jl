## Run all (alt+enter)
include(relpath(pwd(), @__DIR__) * "/src/HasegawaWakatini.jl")

## Run "Gyro-Bohm model"
#domain = Domain(128, 128, 32, 32, anti_aliased=true)
domain = Domain(256, 256, 48, 48, anti_aliased=true)
ic = initial_condition_linear_stability(domain, 1e-3)

# Linear operator
function L(u, d, p, t)
    D_n = p["D"] .* diffusion(u, d)
    D_Ω = p["D"] .* diffusion(u, d)
    cat(D_n, D_Ω, dims=3)
end

# Non-linear operator, linearized
function N(u, d, p, t)
    n = @view u[:, :, 1]
    Ω = @view u[:, :, 2]
    ϕ = solvePhi(Ω, d)

    dn = -poissonBracket(ϕ, n, d)
    dn .-= (1 - p["g"]) * diffY(ϕ, d)
    dn .-= p["g"] * diffY(n, d)
    dn .+= p["sigma"] * ϕ

    dΩ = -poissonBracket(ϕ, Ω, d)
    dΩ .-= p["g"] * diffY(n, d)
    dΩ .+= p["sigma"] * ϕ
    return cat(dn, dΩ, dims=3)
end

# Parameters
parameters = Dict(
    "D" => 1e-2,
    "g" => 1e-1,
    "sigma" => 2e-2,
)

t_span = [0, 500_000]

prob = SpectralODEProblem(L, N, domain, ic, t_span, p=parameters, dt=2e-3)

# Diagnostics
diagnostics = [
    ProgressDiagnostic(10000),
    ProbeAllDiagnostic([(x, 0) for x in range(-24, 19.6, 10)], N=10),
    # PlotDensityDiagnostic(500),
    # RadialFluxDiagnostic(100),
    # KineticEnergyDiagnostic(100),
    # PotentialEnergyDiagnostic(100),
    # EnstropyEnergyDiagnostic(100),
    # GetLogModeDiagnostic(500, :ky),
    # CFLDiagnostic(500),
    # RadialPotentialEnergySpectraDiagnostic(500),
    # PoloidalPotentialEnergySpectraDiagnostic(500),
    # RadialKineticEnergySpectraDiagnostic(500),
    # PoloidalKineticEnergySpectraDiagnostic(500),
]

# Output
cd(relpath(@__DIR__, pwd()))
output = Output(prob, 1001, diagnostics, "../output/gyro-bohm=2e-2 CUDA.h5",
    simulation_name=:parameters, store_locally=false)

FFTW.set_num_threads(16)

## Solve and plot
sol = spectral_solve(prob, MSS3(), output, resume=true)

send_mail("sigma=2e-2 finnished, go analyse the data!")
close(output.file)