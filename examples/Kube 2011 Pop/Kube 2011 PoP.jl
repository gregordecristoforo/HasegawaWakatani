## Run all (alt+enter)
using HasegawaWakatani

## Run scheme test for Burgers equation
domain = Domain(1024, 1024, 50, 50, dealiased=false)
#domain = Domain(256, 256, 50, 50, dealiased=false)
u0 = log.(gaussian.(domain.x', domain.y, A=1e5, B=1, l=1))

# Linear operator
function L(u, d, p, t)
    D_η = p["kappa"] * laplacian(u, d)
    D_Ω = p["nu"] * laplacian(u, d)
    cat(D_η, D_Ω, dims=3)
end

# Non-linear operator
function N(u, d, p, t)
    η = @view u[:, :, 1]
    Ω = @view u[:, :, 2]
    ϕ = solve_phi(Ω, d)
    dη = -poisson_bracket(ϕ, η, d)
    dη += p["kappa"] * quadratic_term(diff_x(η, d), diff_x(η, d), d)
    dη += p["kappa"] * quadratic_term(diff_y(η, d), diff_y(η, d), d)
    dΩ = -poisson_bracket(ϕ, Ω, d)
    dΩ -= diff_y(η, d)
    return cat(dη, dΩ, dims=3)
end

# Parameters
parameters = Dict(
    "nu" => 1e-3,
    "kappa" => 1e-3
)

# Time interval
t_span = [0, 16]

# Speed up simulation
FFTW.set_num_threads(16)

# Inverse transform
function inverse_transformation(u)
    @. u[:, :, 1] = exp(u[:, :, 1]) - 1
end

# The problem
prob = SpectralODEProblem(L, N, [u0;;; zero(u0)], domain, t_span, p=parameters, dt=1e-3)

# Array of diagnostics want
diagnostics = [
    RadialCOMDiagnostic(1),
    ProgressDiagnostic(100),
    PlotDensityDiagnostic(1000)
]

# Folder path
cd(relpath(@__DIR__, pwd()))

# The output
output = Output(prob, 201, diagnostics, "Kube 2011 Pop.h5", physical_transform=inverse_transformation, store_locally=false)

## Solve and plot
sol = spectral_solve(prob, MSS3(), output)

plot(output.simulation["RadialCOMDiagnostic/t"][1:200], output.simulation["RadialCOMDiagnostic/data"][2, 1:200])

## Recreate max velocity plot Kube 2011
tends = logspace(2, 0.30, 22)
tends[8] = 10^(1.131939295)
tends[9:end] = logspace(0.771485063978876, 0.30, 14)
dts = tends / 10000
amplitudes = logspace(-2, 5, 22)
max_velocities = similar(amplitudes)
velocities = Vector{AbstractArray}(undef, length(amplitudes))

parameters = Dict(
    "nu" => 1e-3,
    "kappa" => 1e-3
)

for (i, A) in enumerate(amplitudes[12:end])
    i = i + 11
    # Update initial initial_condition
    u0 = log.(gaussian.(domain.x', domain.y, A=A, B=1, l=1))
    # Update problem 
    prob = SpectralODEProblem(L, N, [u0;;; zero(u0)], domain, [0, tends[i]], p=parameters, dt=dts[i],
        inverse_transformation=inverse_transformation)
    # Reset diagnostics
    diagnostics = [RadialCOMDiagnostic(10), ProgressDiagnostic(100), PlotDensityDiagnostic(1000),]
    # Update output
    output = Output(prob, 21, diagnostics, "output/Kube 2011 Pop 1024x1024 max vel.h5")
    # Solve 
    sol = spectral_solve(prob, MSS3(), output)
    # Extract velocity
    velocities[i] = extract_diagnostic(sol.diagnostics[1].data)
    # Determine max velocity
    max_velocities[i] = maximum(velocities[i][2, :])
end

# Ploting
ytickslabels = [""; ""; L"10^{-1}"; fill("", 8); L"10^{0}"; fill("", 8); L"10^{1}"]
scatter(amplitudes, max_velocities, xaxis=:log, yaxis=:log, marker=:circle, linestyle=:dash, label="")
plot!(amplitudes[1:7], 0.83 * amplitudes[1:7] .^ 0.5, label=L"0.83(\Delta n/N)^{0.5}", ylabel="max " * L"V", xlabel=L"\Delta n/N",
    xticks=amplitudes[1:3:end], legend=:bottomright, yticks=([0.08; 0.09; 0.1:0.1:1; 2:1:10], ytickslabels), ylim=[max_velocities[1] - 0.01, 10])
savefig("blob velocities log(n).pdf")

## Backup
max_velocities = [0.08144987557214223, 0.12116171785371747, 0.17908891760090645, 0.2627304330198832, 0.3812298636057552,
    0.5436405787806564, 0.7554935875115005, 1.0158889263024147, 1.317316393304872, 1.6476427382449557,
    1.993949292800375, 2.345936231222947, 2.6969337556930855, 3.043242064725629, 3.38310106072764,
    3.7158630435705096, 4.04143715620406, 4.3600155702429, 4.6719064564576, 4.97746001501175,
    5.277028416262867, 5.570952278649872]

## Can use one of the many .h5 files and groups to reproduce other plots from section 2 A

tends = [30, 15, 7]
dts = tends / 10_000
amplitudes = [0.1, 1, 10]
max_velocities = similar(amplitudes)
velocities = Vector{Matrix}(undef, length(amplitudes))

for (i, A) in enumerate(amplitudes)
    # Update initial initial_condition
    u0 = gaussian.(domain.x', domain.y, A=A, B=0, l=1)
    # Update problem 
    prob = SpectralODEProblem(L, N, [u0;;; zero(u0)], domain, [0, tends[i]], p=parameters, dt=dts[i])
    # Reset diagnostics
    diagnostics = [RadialCOMDiagnostic(1), ProgressDiagnostic(100), PlotDensityDiagnostic(1000),]
    # Update output
    output = Output(prob, 21, diagnostics, "Kube finale.h5", store_locally=false, simulation_name=string(A))
    # Solve 
    sol = spectral_solve(prob, MSS3(), output, resume=false)
    # Extract velocity
    velocities[i] = sol.simulation["RadialCOMDiagnostic/data"][:, :]
    # Determine max velocity
    max_velocities[i] = maximum(velocities[i][2, :])

    CUDA.pool_status()
end

data = sol.simulation["RadialCOMDiagnostic/data"][2, :]

fid = h5open("Kube finale.h5")
sim = fid["1"]
data = sim["RadialCOMDiagnostic/data"][2, :]
plot(sim["RadialCOMDiagnostic/t"][:], data, aspect_ratio=:auto)

using JLD
jldopen("blob evolution kube.jld", "w") do file
    g = create_group(file, "data")
    g["10"] = fid["10"]["RadialCOMDiagnostic/data"][2, 1:end]
    g["1"] = fid["1"]["RadialCOMDiagnostic/data"][2, 1:end]
    g["0.1"] = fid["0.1"]["RadialCOMDiagnostic/data"][2, 1:end]
    g["t1"] = fid["10"]["RadialCOMDiagnostic/t"][:]
    g["t2"] = fid["1"]["RadialCOMDiagnostic/t"][:]
    g["t3"] = fid["0.1"]["RadialCOMDiagnostic/t"][:]
end

heatmap(sim["fields"][:, :, 1, end])