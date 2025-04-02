## Run all (alt+enter)
include(relpath(pwd(), @__DIR__) * "/src/HasegawaWakatini.jl")

## Run Hasegawa Wakatani simulations
domain = Domain(128, 128, 2 * pi / 0.15, 2 * pi / 0.15, anti_aliased=true)

#Random initial conditions
ic = initial_condition_linear_stability(domain, 10^-1)

# Linear operator
function L(u, d, p, t)
    D_n = p["D_n"] .* hyper_diffusion(u, d) .- p["C"] * u
    D_Ω = p["D_Ω"] .* hyper_diffusion(u, d)
    [D_n;;; D_Ω]
end

# Non-linear terms
function N(u, d, p, t)
    n = @view u[:, :, 1]
    Ω = @view u[:, :, 2]
    ϕ = solvePhi(Ω, d)
    dn = -poissonBracket(ϕ, n, d)
    dn .-= p["kappa"] .* diffY(ϕ, d)
    dn .+= p["C"] .* (ϕ)
    dΩ = -poissonBracket(ϕ, Ω, d)
    dΩ .+= p["C"] .* (ϕ .- n)
    return [dn;;; dΩ]
end

# Parameters
parameters = Dict(
    "D_n" => 1e-2,
    "D_Ω" => 1e-2,
    "kappa" => 1,
    "C" => 5,
)

t_span = [0, 20000]

prob = SpectralODEProblem(L, N, domain, ic, t_span, p=parameters, dt=1e-3)

# Diagnostics
diagnostics = [
    ProgressDiagnostic(1000),
    ProbeDensityDiagnostic((0, 0), N=1000),
    #PlotDensityDiagnostic(500),
    RadialFluxDiagnostic(500),
    KineticEnergyDiagnostic(500),
    PotentialEnergyDiagnostic(500),
    EnstropyEnergyDiagnostic(500),
    GetLogModeDiagnostic(500, :ky),
    CFLDiagnostic(500),
    RadialPotentialEnergySpectraDiagnostic(500),
    PoloidalPotentialEnergySpectraDiagnostic(500),
    RadialKineticEnergySpectraDiagnostic(500),
    PoloidalKineticEnergySpectraDiagnostic(500),
]

# Output
cd("tests/HW")
output = Output(prob, 201, diagnostics, "output/Hasegawa-Wakatini april first C=5.h5")

FFTW.set_num_threads(16)

# Solve and plot
sol = spectral_solve(prob, MSS3(), output; resume=true)

#data = sol.simulation["Kinetic energy integral/data"][:]
#t = sol.simulation
#plot(data)

#plot(sol.diagnostics[end-3].t, sol.diagnostics[end-3].data)

## Parameter scan
#C_values = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]

#for C in C_values
#    println("Started simulation for C = $(C)!")
#    prob.p["C"] = C
#    #prob = SpectralODEProblem(L, N, domain, ic, t_span, p=parameters, dt=1e-2)
#    output = Output(prob, 201, diagnostics, "output/Hasegawa-Wakatini april first.h5")
#    sol = spectral_solve(prob, MSS3(), output)
#end
