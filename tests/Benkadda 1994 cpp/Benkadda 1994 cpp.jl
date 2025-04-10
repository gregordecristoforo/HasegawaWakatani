## Run all (alt+enter)
include(relpath(pwd(), @__DIR__)*"/src/HasegawaWakatini.jl")

## Run Benkadda simulations
domain = Domain(256, 256, 128, 128, anti_aliased=true)
# TODO find initial condition
ic = initial_condition_linear_stability(domain,1e-3)

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
    dn .-= (p["kappa"] - p["g"]) * diffY(ϕ, d)
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
    "ν" => 1e-2,
    "g" => 1e-1,
    "σ" => 1e-3,
    "kappa" => 1.0 #? Not used
)

t_span = [0, 1000]

prob = SpectralODEProblem(L,N, domain, ic, t_span, p=parameters, dt=1e-3)

# Diagnostics
diagnostics = [
    ProgressDiagnostic(1000),
    ProbeDensityDiagnostic((0, 0), N=100),
    PlotDensityDiagnostic(1000),
    #RadialFluxDiagnostic(50),
    #KineticEnergyDiagnostic(50),
    #PotentialEnergyDiagnostic(50),
    #EnstropyEnergyDiagnostic(50),
    #GetLogModeDiagnostic(50, :ky),
    #CFLDiagnostic(50),
    #RadialPotentialEnergySpectraDiagnostic(50),
    #PoloidalPotentialEnergySpectraDiagnostic(50),
    #RadialKineticEnergySpectraDiagnostic(50),
    #PoloidalKineticEnergySpectraDiagnostic(50),
]

# Output
cd("tests/Benkadda 1994 cpp")
output = Output(prob, 1001, diagnostics, "output/benkadda april tenth.h5", simulation_name=:parameters)

FFTW.set_num_threads(16)

## Solve and plot
sol = spectral_solve(prob, MSS3(), output, resume=true)

data = sol.simulation["fields"][:,:,:,:]
t = sol.simulation["t"][:]
default(legend=false)
anim = @animate for i in axes(data, 4)
    heatmap(data[:, :, 1, i], aspect_ratio=:equal, xaxis=L"x", yaxis=L"y", title=L"n(t="*"$(round(t[i], digits=0)))")
end
gif(anim, "benkadda.gif", fps = 20)

send_mail("Benkadda simulation finnished!", attachment="benkadda.gif")
close(output.file)