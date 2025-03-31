## Run all (alt+enter)
include("../src/HasegawaWakatini.jl")

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
    dη = -(1 - p["g"]) * diffY(ϕ, d)
    dη -= p["g"] * diffY(η, d)
    #dη .+= p["D_n"] - p["sigma_n"]
    dη -= p["sigma_n"] * ϕ
    dη -= 2 * p["D_n"] * p["kappa"] * diffX(η, d)
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
t_span = [0, 3600]

# The problem
prob = SpectralODEProblem(L, N, domain, ic, t_span, p=parameters, dt=1e-3)

# Array of diagnostics want
diagnostics = [
    #ProbeDensityDiagnostic([(5, 0), (8.5, 0), (11.25, 0), (14.375, 0)], N=10),
    #RadialCOMDiagnostic(),
    ProgressDiagnostic(1000),
    #CFLDiagnostic(),
    #RadialCFLDiagnostic(100),
    PlotDensityDiagnostic(10000),
    #GetModeDiagnostic(100),
    GetLogModeDiagnostic(100),
]

# The output
output = Output(prob, 1001, diagnostics) #progressDiagnostic

## Solve and plot
sol = spectral_solve(prob, MSS3(), output)

## Recreate Garcia et al. plots
display(heatmap(sol.u[5][:, :, 1], levels=10, aspect_ratio=:equal))
display(heatmap(sol.u[10][:, :, 1], levels=10, aspect_ratio=:equal))
display(heatmap(sol.u[15][:, :, 1], levels=10, aspect_ratio=:equal))
display(heatmap(sol.u[end][:, :, 1], levels=10, aspect_ratio=:equal))
display(heatmap(sol.u[5][:, :, 2], levels=10, aspect_ratio=:equal, color=:jet))
display(heatmap(sol.u[10][:, :, 2], levels=10, aspect_ratio=:equal, color=:jet))
display(heatmap(sol.u[15][:, :, 2], levels=10, aspect_ratio=:equal, color=:jet))
display(heatmap(sol.u[end][:, :, 2], levels=10, aspect_ratio=:equal, color=:jet))

## Save data
using JLD
save("linearstability.jld", "data", sol.u)

## Analyze data
data = Array(reshape(reduce(hcat, sol.u), size(sol.u[1])..., length(sol.u)))
data = Matrix{Number}(data')

# Unreliable atm
#data = Array(reshape(reduce(hcat, sol.u), size(sol.u[1])..., length(sol.u)))
kappa = sqrt(parameters["sigma_n"] / parameters["D_n"])
n0 = initial_condition(exponential_background, domain, kappa=kappa)
data_hat = zeros(ComplexF64, size(sol.u[1])..., length(sol.u))
for i in eachindex(sol.u)
    data_hat[:, :, 1, i] = fft(sol.u[i][:, :, 1])#fft(n0.*sol.u[i][:, :, 1])
    data_hat[:, :, 2, i] = fft(sol.u[i][:, :, 2])
end

plot(log.(abs.(data_hat[60, 60, 1, :])))
plot(domain.x, n0[1, :] + 1e4 * sol.u[end][1, :, 1])

plot(log.(abs.(data_hat[14, 14, 1, :])))

# Check if can get growth rate from taking difference between two last data points
gamma = log.(abs.(data_hat[:, :, 1, end])) .- log.(abs.(data_hat[:, :, 1, end-1]))
plot(fftshift(domain.kx)[end-62:end], fftshift(real(diag(gamma)))[end-62:end], xaxis=:log)
ylims!(0, 0.003)

fftshift(domain.kx)[end-62]


gamma = sol.diagnostics[3].data[end][:,1] - sol.diagnostics[3].data[end-1][:,1]
plot(gamma, xaxis=:log)

for i in eachindex(sol.diagnostics[3].data)
    display(plot(sol.diagnostics[3].data[i][:,1]))
end

mapreduce(permutedims, hcat, sol.diagnostics[3].data)


sol.diagnostics[3].data[1]

Array(reshape(reduce(hcat, output.u), size(output.u[1])..., length(output.u)))

a = reshape(reduce(hcat, sol.diagnostics[3].data), (64,2,36001))

plot(domain.kx[2:64], a[2:64,1,end]-a[2:64,1,end-1], xaxis=:log)

gamma = a[1:64,1,end]-a[1:64,1,end-1]
gamma[4]