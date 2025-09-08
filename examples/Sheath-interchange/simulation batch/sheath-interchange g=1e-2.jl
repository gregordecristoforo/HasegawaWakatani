## Run all (alt+enter)
include(relpath(pwd(), @__DIR__) * "/src/HasegawaWakatini.jl")

## Run scheme test for Burgers equation
domain = Domain(128, 128, 54, 54, anti_aliased=true)
ic = initial_condition_linear_stability(domain, 1e-3)
#ic[:, :, 1] .+= 1

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
    ϕ = solve_phi(Ω, d)

    dn = -poisson_bracket(ϕ, n, d)
    dn .-= (p["kappa"] - p["g"]) * diff_y(ϕ, d)
    dn .-= p["g"] * diff_y(n, d)
    dn .-= p["sigma_n"] * n

    dΩ = -poisson_bracket(ϕ, Ω, d)
    dΩ .-= p["g"] * diff_y(n, d)
    dΩ .+= p["sigma_Ω"] * ϕ
    return [dn;;; dΩ]
end

function N(u, d, p, t)
    n = @view u[:, :, 1]
    Ω = @view u[:, :, 2]
    ϕ = solve_phi(Ω, d)

    dn = -poisson_bracket(ϕ, n, d)
    dn .-= (p["kappa"] - p["g"]) * diff_y(ϕ, d)#.+= p["g"]* quadratic_term(n, diff_y(ϕ, d), d)
    #dn .-= 2 * p["kappa"] * p["D_n"] * diff_x(n, d)
    #dn .+= p["D_n"] * quadratic_term(diff_x(n, d), diff_x(n, d), d)
    #dn .+= p["D_n"] * quadratic_term(diff_y(n, d), diff_y(n, d), d)
    dn .-= p["g"] * diff_y(n, d)
    dn .+= p["sigma_n"] * ϕ #* spectral_exp(-ϕ, d) #quadratic_term(n, spectral_exp(-ϕ, d), d)

    dΩ = -poisson_bracket(ϕ, Ω, d)
    dΩ .-= p["g"] * diff_y(n, d) # .-= p["g"] * diff_y(spectral_log(n, d), d)
    #dΩ .+= p["sigma_Ω"]*ϕ #* spectral_expm1(-ϕ, d)
    return [dn;;; dΩ]
end

# Parameters
parameters = Dict(
    "D_Ω" => 1e-2,
    "D_n" => 1e-2,
    "g" => 1e-2,
    "sigma_Ω" => 1e-3,
    "sigma_n" => 1e-3,
    "kappa" => sqrt(1e-1),
    "N" => 1.0
)

t_span = [0, 500_000]

prob = SpectralODEProblem(L, N, domain, ic, t_span, p=parameters, dt=1e-2)

# Diagnostics
diagnostics = [
    ProgressDiagnostic(10000),
    ProbeAllDiagnostic([(x, 0) for x in LinRange(-40, 50, 10)], N=10),
    PlotDensityDiagnostic(500),
    RadialFluxDiagnostic(100),
    KineticEnergyDiagnostic(100),
    PotentialEnergyDiagnostic(100),
    EnstropyEnergyDiagnostic(100),
    GetLogModeDiagnostic(500, :ky),
    CFLDiagnostic(500),
    RadialPotentialEnergySpectraDiagnostic(500),
    PoloidalPotentialEnergySpectraDiagnostic(500),
    RadialKineticEnergySpectraDiagnostic(500),
    PoloidalKineticEnergySpectraDiagnostic(500),
]

# Output
cd(relpath(@__DIR__, pwd()))
output = Output(prob, 1001, diagnostics, "../output/sheath-interchange g=1e-2.h5",
    simulation_name=:parameters, store_locally=false)

FFTW.set_num_threads(16)

## Solve and plot
sol = spectral_solve(prob, MSS3(), output, resume=false)

send_mail("g=1e-2 finnished, go analyse the data!")
close(output.file)