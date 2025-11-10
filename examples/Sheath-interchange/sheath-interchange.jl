## Run all (alt+enter)
using HasegawaWakatani
using CUDA

## Run scheme test for Burgers equation
domain = Domain(128, 128; Lx=100, Ly=100, MemoryType=CuArray, precision=Float32)
ic = initial_condition(random_crossphased, domain; value=1e-3)

# Linear operator
function Linear(du, u, operators, p, t)
    @unpack laplacian = operators
    η, Ω = eachslice(u; dims=3)
    dη, dΩ = eachslice(du; dims=3)
    @unpack ν, μ = p
    dη .= ν * laplacian(η)
    dΩ .= μ * laplacian(Ω)
end

# Non-linear operator, linearized
function NonLinear(du, u, operators, p, t)
    @unpack solve_phi, poisson_bracket, diff_y = operators
    η, Ω = eachslice(u; dims=3)
    dη, dΩ = eachslice(du; dims=3)
    @unpack κ, g, σ = p
    ϕ = solve_phi(Ω)

    dη .= poisson_bracket(η, ϕ) - (κ - g) * diff_y(ϕ) - g * diff_y(η) + σ * ϕ
    dΩ .= poisson_bracket(Ω, ϕ) - g * diff_y(η) + σ * ϕ
end

# Non-linear operator, fully non-linear
function NonLinear(du, u, operators, p, t)
    @unpack solve_phi, poisson_bracket, diff_x, diff_y = operators
    @unpack quadratic_term, spectral_exp, spectral_expm1, spectral_constant = operators
    η, Ω = eachslice(u; dims=3)
    dη, dΩ = eachslice(du; dims=3)
    @unpack κ, g, σ, ν = p
    ϕ = solve_phi(Ω)

    dη .= poisson_bracket(η, ϕ) - (κ - g) * diff_y(ϕ) - 2 * ν * κ * diff_x(η) +
          spectral_constant(ν * κ .^ 2) + ν * quadratic_term(diff_x(η), diff_x(η)) +
          ν * quadratic_term(diff_y(η), diff_y(η)) + σ * spectral_exp(-ϕ) - g * diff_y(η)
    dΩ .= poisson_bracket(Ω, ϕ) - g * diff_y(η) - σ * spectral_expm1(-ϕ)
end

# Parameters
parameters = (κ=1e-2, g=1e-3, σ=1e-3, ν=1e-2, μ=1e-4)

# Time interval
tspan = [0.0, 5000000.0]

# Diagnostics
diagnostics = @diagnostics [
    progress(; stride=1000),
    probe_all(; positions=[(x, 0) for x in LinRange(-50, 40, 10)], stride=100),
    plot_density(; stride=1000),
    radial_flux(; stride=50),
    kinetic_energy_integral(; stride=50),
    potential_energy_integral(; stride=50),
    enstropy_energy_integral(; stride=50),
    get_log_modes(; stride=50, axis=:diag),
    cfl(; stride=5000, silent=true),
    sample_density(; storage_limit="1 GB"),
    sample_vorticity(; storage_limit="1 GB"),
    sample_potential(; storage_limit="1 GB")
    #potential_energy_spectrum(; spectrum=:radial, stride=50),
    #potential_energy_spectrum(; spectrum=:poloidal, stride=50),
    #kinetic_energy_spectrum(; spectrum=:radial, stride=50),
    #kinetic_energy_spectrum(; spectrum=:poloidal, stride=50)
]

# Collection of specifications defining the problem to be solved
prob = SpectralODEProblem(Linear, NonLinear, ic, domain, tspan; p=parameters, dt=1e-1,
                          operators=:all, diagnostics=diagnostics)

# Inverse transform
inverse_transformation!(u) = @. u[:, :, 1] = exp(u[:, :, 1]) - 1

# Output
output_file_name = joinpath(@__DIR__, "output", "sheath-interchange long time series.h5")
output = Output(prob; filename=output_file_name, simulation_name=:parameters,
                physical_transform=inverse_transformation!, storage_limit="1 GB",
                store_locally=false)

## Solve and plot
sol = spectral_solve(prob, MSS3(), output; resume=false)

using SMTPClient
send_mail("Long time series simulation finnished!"; attachment="benkadda.gif")
close(output)