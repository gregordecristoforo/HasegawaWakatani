## Run all (alt+enter)
include("../../src/HasegawaWakatini.jl")

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
    "C" => 1,
)

t_span = [0, 2000]

prob = SpectralODEProblem(L, N, domain, ic, t_span, p=parameters, dt=1e-2)

# Diagnostics
diagnostics = [
    ProgressDiagnostic(1000),
    ProbeDensityDiagnostic((0, 0), N=1000),
    PlotDensityDiagnostic(500),
    RadialFluxDiagnostic(500),
    KineticEnergyDiagnostic(500),
    PotentialEnergyDiagnostic(500),
    GetLogModeDiagnostic(500, :ky),
    CFLDiagnostic(500),
]

# Output
cd("tests/HW")
output = Output(prob, 201, diagnostics, "debuging things.h5") 

FFTW.set_num_threads(16)

# Solve and plot
sol = spectral_solve(prob, MSS3(), output; resume=true)

plot(sol.diagnostics[end-3].t, sol.diagnostics[end-3].data)



## Parameter scan
# C_values = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]

# for C in C_values
#     println("Started simulation for C = $(C)!")
#     prob.p["C"] = C
#     #prob = SpectralODEProblem(L, N, domain, ic, t_span, p=parameters, dt=1e-2)
#     output = Output(prob, 201, diagnostics, "Hasegawa-Wakatani C weekend scan.h5")
#     sol = spectral_solve(prob, MSS3(), output)
# end

u_hat = transform(sol.u[end], domain.transform.FT)

n_hat = u_hat[:,:,1]
omega_hat = u_hat[:,:,2]


plot([sum(abs.(irfft(n_hat[:,i], 128))) for i in 1:128])
plot([sum(abs.(ifft(n_hat[i,:]))) for i in 1:65])

test = fft(n_hat[:,:])

plot(abs.(test))

sum(abs.(n_hat[:,:]))/(domain.Lx*domain.Ly)
sol.diagnostics[4].data[end]


sum(-domain.SC.Laplacian.*abs.(u_hat[:,:,2]))


(sum(abs.(n_hat[1:end,:])) - 0.5*sum(abs.(n_hat[1,:])))/(domain.Nx*domain.Ny)
sol.diagnostics[5].data[end]

1/2*sum(sol.u[end][:,:,1].^2)


test = fft(sol.u[end][:,:,2])

1/2*sum(abs.(test).^2)/(128*128) - sol.diagnostics[5].data[end]


# Calculate density energy using Parsevals theorem:
E_k = abs.(n_hat).^2 
(sum(E_k) - 0.5*sum(E_k[1,:]))/(domain.Nx*domain.Ny)
sol.diagnostics[5].data[end]

# Calculate kinetic energy using Parsevals theorem:
E_k = abs.(omega_hat).^2 #(domain.kx'.^2 .+ domain.ky.^2).*
(sum(E_k) - 0.5*sum(E_k[1,:]))/(domain.Nx*domain.Ny) - sol.diagnostics[4].data[end]

kinetic_energy_integral(sol.u[end], prob, 1)

phi_hat = solvePhi(u_hat[:,:,2], domain)

contourf(domain.transform.iFT*-diffY(phi_hat, domain))

radii = -domain.SC.Laplacian
dk = 0.5*0.15

radii.÷dk

radiidx = round.(Int, radii/dk) .+ 1

bins = zeros(maximum(radiidx))

for i in eachindex(n_hat)
    bins[radiidx[i]] += abs(n_hat[i]).^2
end

plot(bins .+ 1e-50, xaxis=:log, yaxis=:log)

bins = 0:dk:maximum(radii)+dk
energy = zeros(length(bins))

heatmap((bins[1] .<= radii .<= bins[2]))
heatmap((bins[2] .<= radii .<= bins[3]))
heatmap((bins[3] .<= radii .<= bins[4]))
heatmap((bins[4] .<= radii .<= bins[5]))
heatmap((bins[5] .<= radii .<= bins[6]))
heatmap((bins[6] .<= radii .<= bins[7]))
heatmap((bins[7] .<= radii .<= bins[8]))
heatmap((bins[8] .<= radii .<= bins[9]))
heatmap((bins[end-10] .<= radii .<= bins[end]))
bins[end-3]
radii[65, 65]
maximum(radii)

maximum(sqrt(radii))
maximum(domain.kx).^2