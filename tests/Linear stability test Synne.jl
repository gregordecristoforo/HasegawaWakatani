## Run all (alt+enter)
include("../src/HasagawaWakatini.jl")

## Run scheme test for Burgers equation
#domain = Domain(512, 512, 200, 100, anti_aliased=false)
domain = Domain(128, 128, 500, 500, anti_aliased=false)
#domain = Domain(1024, 1024, 50, 50, anti_aliased=false)
#u0 = gaussian.(domain.x', domain.y, A=0.1, B=1, l=10)
#u0 = all_modes_with_random_phase(domain, 1e-6)
#phi0 = all_modes_with_random_phase(domain, 1e-6)
#Omega0 = domain.transform.iFT*diffusion(domain.transform.FT*phi0,domain)

ic = initial_condition_linear_stability(domain)
#ic = [u0;;; zero(u0)]

# Linear operator (May not be static actually)
function L(u, d, p, t)
    D_η = p["D_n"] * diffusion(u, d) #.- p["g"]*diffY(u,d)
    D_Ω = p["D_Omega"] * diffusion(u, d)
    [D_η;;; D_Ω]
end

# Non-linear operator
function N(u, d, p, t)
    η = u[:, :, 1]
    Ω = u[:, :, 2]
    ϕ = solvePhi(Ω, d)
    dη = -(p["kappa"] - p["g"]) * diffY(ϕ, d)
    dη -= p["g"] * diffY(η, d)
    dη -= p["sigma_n"] * η
    dη -= p["sigma_n"] * ϕ
    #dη -= p["sigma_n"] * η
    dΩ = -p["g"] * diffY(η, d)
    dΩ += p["sigma_Omega"] * ϕ
    return [dη;;; dΩ]
end

# Parameters
parameters = Dict(
    "D_Omega" => 1e-2,
    "D_n" => 1e-2,
    "g" => 1e-3,
    "sigma_Omega" => 1e-5,
    "sigma_n" => 1e-5,
    "kappa" => sqrt(1e-3)
)

# Time interval
t_span = [0, 500]

# The problem
prob = SpectralODEProblem(L, N, domain, ic, t_span, p=parameters, dt=1e-3)

# Array of diagnostics want
diagnostics = [
    #ProbeDensityDiagnostic([(5, 0), (8.5, 0), (11.25, 0), (14.375, 0)], N=10),
    #RadialCOMDiagnostic(),
    ProgressDiagnostic(100),
    #CFLDiagnostic(),
    #RadialCFLDiagnostic(100),
    PlotDensityDiagnostic(1000),
    GetModeDiagnostic(100),
]

# The output
output = Output(prob, 1001, diagnostics) #progressDiagnostic

## Solve and plot
sol = spectral_solve(prob, MSS3(), output)

# ------------------ Mode analysis ---------------------------------------------------------

## Analyze data
data = Array(reshape(reduce(hcat, sol.u), size(sol.u[1])..., length(sol.u)))
data = Matrix{Number}(data')

# Unreliable atm
#data = Array(reshape(reduce(hcat, sol.u), size(sol.u[1])..., length(sol.u)))
data_hat = zeros(ComplexF64, size(sol.u[1])..., length(sol.u))
for i in eachindex(sol.u)
    data_hat[:, :, 1, i] = fft(sol.u[i][:, :, 1])
    data_hat[:, :, 2, i] = fft(sol.u[i][:, :, 2])
end

wavenumber_overtime = zeros(1001, domain.Nx)
for i in eachindex(domain.kx)
    wavenumber_overtime[:, i] = log.(abs.(data_hat[i, i, 1, :])) .- log.(abs.(data_hat[i, i, 1, 10]))
end


for i in eachindex(domain.kx)
    display(plot(wavenumber_overtime[:,i], title="k_$i = $(domain.kx[i])"))
end


# Check if can get growth rate from taking difference between two last data points
gamma = log.(abs.(data_hat[:, :, 1, end])) .- log.(abs.(data_hat[:, :, 1, end-1]))
plot(fftshift(domain.kx)[end-62:end], fftshift(real(diag(gamma)))[end-62:end], xaxis=:log)
ylims!(0, 0.003)

fftshift(domain.kx)[end-62]
