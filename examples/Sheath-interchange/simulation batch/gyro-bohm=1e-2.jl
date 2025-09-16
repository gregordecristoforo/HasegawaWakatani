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
    ϕ = solve_phi(Ω, d)

    dn = -poisson_bracket(ϕ, n, d)
    dn .-= (1 - p["g"]) * diff_y(ϕ, d)
    dn .-= p["g"] * diff_y(n, d)
    dn .+= p["sigma"] * ϕ

    dΩ = -poisson_bracket(ϕ, Ω, d)
    dΩ .-= p["g"] * diff_y(n, d)
    dΩ .+= p["sigma"] * ϕ
    return cat(dn, dΩ, dims=3)
end

# Parameters
parameters = Dict(
    "D" => 1e-2,
    "g" => 1e-1,
    "sigma" => 1e-2,
)

t_span = [0, 100]

prob = SpectralODEProblem(L, N, domain, ic, t_span, p=parameters, dt=1e-3, remove_modes=remove_zonal_modes!)

# Diagnostics
diagnostics = [
    ProgressDiagnostic(10000),
    ProbeAllDiagnostic([(x, 0) for x in range(-24, 19.2, 10)], N=10),
    PlotDensityDiagnostic(500),
    # RadialFluxDiagnostic(100),
    # KineticEnergyDiagnostic(100),
    # PotentialEnergyDiagnostic(100),
    # EnstropyEnergyDiagnostic(100),
    # GetLogModeDiagnostic(500, :ky),
    CFLDiagnostic(500),
    # RadialPotentialEnergySpectraDiagnostic(500),
    PoloidalPotentialEnergySpectraDiagnostic(500),
    RadialKineticEnergySpectraDiagnostic(500),
    PoloidalKineticEnergySpectraDiagnostic(500),
]

# Output
cd(relpath(@__DIR__, pwd()))
output = Output(prob, 1001, diagnostics, "../output/debug.h5",
    simulation_name="linear growth 0.012", store_locally=false)

FFTW.set_num_threads(16)

## Solve and plot
sol = spectral_solve(prob, MSS3(), output, resume=false)

send_mail("sigma=1e-2 finnished, go analyse the data!")
close(output)


fid = h5open("../output/debug.h5")
sim = fid["linear growth 0.012"]

data = sim["All probe/data"][:, :, 1:95_400]

n = data[:, 1, :]
Ω = data[:, 2, :]
ϕ = data[:, 3, :]
vx = data[:, 4, :]
Γ = data[:, 5, :]


plot(ϕ[:, 10000:end]', aspect_ratio=:auto)

using Statistics
plot(mean(n[:, :]', dims=1)', aspect_ratio=:auto)


n_n = (n .- mean(n, dims=1)) ./ std(n, dims=1)
Ω_n = (Ω .- mean(Ω, dims=1)) ./ std(Ω, dims=1)
ϕ_n = (ϕ .- mean(ϕ, dims=1)) ./ std(ϕ, dims=1)
vx_n = (vx .- mean(vx, dims=1)) ./ std(vx, dims=1)
Γ_n = (Γ .- mean(Γ, dims=1)) ./ std(Γ, dims=1)
histogram(n_n', aspect_ratio=:auto, yaxis=:log10)
histogram(Ω_n', aspect_ratio=:auto, yaxis=:log10)
histogram(ϕ_n', aspect_ratio=:auto, yaxis=:log10)
histogram(vx_n', aspect_ratio=:auto, yaxis=:log10)
histogram(Γ_n', aspect_ratio=:auto, yaxis=:log10)

u_hat = sim["cache_backup/u0"][:, :, :]
u_reduced = copy(u_hat)
u_reduced[60:129, :, :] .= 0

n = domain.transform.iFT * u_reduced[:, :, 1]
heatmap(n)

heatmap(sol.simulation["fields"][:, :, 1, 700])

sim = sol.simulation
for N in 600
    #sim = fid[keys(fid)[1]]
    sigma = read_attribute(sim, "sigma")
    n = sim["fields"][:, :, 1, N]
    t = sim["t"][N]
    heatmap(domain, n, title=L"n(x,y,t=60 \omega_c),\ [\ \sigma=" * string(sigma) * L",\ g=0.1,\ D=0.01\ ]", xlabel=L"x\ [\rho_s]",
        ylabel=L"y\ [\rho_s]", size=[600, 550], margin=0Plots.px, top_margin=-100Plots.px, bottom_margin=-40Plots.px, titlefontsize=12, labelfontsize=10)#, color="black")
    #display(contour!(domain.x, domain.y, n, color=:black))
    display(plot!())
    savefig("streamers" * string(sigma) * ".pdf")
end


heatmap(sol.simulation["fields"][:, :, 1, 500])

u_hat = fft(sol.simulation["fields"][:, :, 1, 500])

plot(abs.(u_hat[1:128, 1]), aspect_ratio=:auto)
plot(abs.(u_hat[1, 1:128]), aspect_ratio=:auto)

abs.(u_hat[1:128, 1])
abs.(u_hat[1, 1:128])

plot(sum(abs.(u_hat[:, :, 1]) .^ 2, dims=1)' / (prob.domain.Ny), aspect_ratio=:auto)


data = sum(abs.(u_hat[:, :, 1]) .^ 2, dims=1)' / (prob.domain.Ny)

plot(data, aspect_ratio=:auto)

plot(fftshift(sum(abs2.(u_hat), dims=1)), aspect_ratio=:auto)

Kx = fftshift(sum(abs2.(u_hat), dims=1))[1, :]
Ky = fftshift(sum(abs2.(u_hat), dims=2))[:, 1]

plot(Kx[129:end], aspect_ratio=:auto)
plot!(Ky[129:end], aspect_ratio=:auto)


heatmap(sol.simulation["fields"][:, :, 1, 1001])
u_hat = fft(sol.simulation["fields"][:, :, 1, 1001])
u_hat[:, 2:end] .= 0 # Show streamers
u_hat[2:end, :] .= 0 # Show zonals
u_hat[1, :] .= 0 # Remove zonals
u_hat[:, 1] .= 0 # Remove streamers
u = ifft(u_hat)
heatmap(real(u))