## Run all (alt+enter)
using HasegawaWakatani
using LaTeXStrings
using Plots

domain = Domain(1024, 1024, 50, 50, precision=Float32)#, mem=CuArray)
u0 = gaussian.(domain.x', domain.y, A=1, B=0, l=1)

# Linear operator
function L(u, d, p, t)
    @unpack κ, ν = p
    θ, Ω = eachslice(u, dims=3)
    D_θ = κ * diffusion(θ, d)
    D_Ω = ν * diffusion(Ω, d)
    cat(D_θ, D_Ω, dims=3)
end

# Non-linear operator
function N(u, d, p, t)
    θ, Ω = eachslice(u, dims=3)
    ϕ = solve_phi(Ω, d)
    dθ = -poisson_bracket(ϕ, θ, d)
    dΩ = -poisson_bracket(ϕ, Ω, d)
    dΩ .-= diff_y(θ, d)
    return cat(dθ, dΩ, dims=3)
end

# Parameters
parameters = (ν=1e-2, κ=1e-2)

# Time interval
tspan = [0.0, 20.0]

# Speed up simulation
#FFTW.set_num_threads(16)

# The problem
prob = SpectralODEProblem(L, N, [u0;;; zero(u0)], domain, tspan, p=parameters, dt=1e-3)

# Array of diagnostics want
diagnostics = [
    ProbeDensityDiagnostic([(5, 0), (8.5, 0), (11.25, 0), (14.375, 0)], N=10),
    RadialCOMDiagnostic(1),
    PlotDensityDiagnostic(100),
    ProgressDiagnostic(100),
    CFLDiagnostic(1),
    PlotDensityDiagnostic(1000),
    PlotVorticityDiagnostic(1000),
    PlotPotentialDiagnostic(1000),
]

# Folder path
cd(relpath(@__DIR__, pwd()))

## The output
output = Output(prob, filename="output/Garcia 2005 PoP.h5", diagnostics=diagnostics,
    step_stride=-1, simulation_name=:parameters, field_storage_limit="0.5 GB",
    store_locally=true)

# Solve and plot
sol = spectral_solve(prob, MSS3(), output, resume=false)

## Recreate Garcia et al. plots Figure 1.
savefig(heatmap(domain, sol.u[6][:, :, 1], levels=10, aspect_ratio=:equal, xlabel=L"x", ylim=[-10, 10], size=(600, 280),
        ylabel=L"y", title=L"n(x, t = " * "$(round(sol.t[6], digits=2)))"), "blob density t=5.pdf")
savefig(heatmap(domain, sol.u[11][:, :, 1], levels=10, aspect_ratio=:equal, xlabel=L"x", ylim=[-10, 10], size=(600, 280),
        ylabel=L"y", title=L"n(x, t = " * "$(round(sol.t[11], digits=2)))"), "blob density t=10.pdf")
savefig(heatmap(domain, sol.u[16][:, :, 1], levels=10, aspect_ratio=:equal, xlabel=L"x", ylim=[-10, 10], size=(600, 280),
        ylabel=L"y", title=L"n(x, t = " * "$(round(sol.t[16], digits=2)))"), "blob density t=15.pdf")
savefig(heatmap(domain, sol.u[end][:, :, 1], levels=10, aspect_ratio=:equal, xlabel=L"x", ylim=[-10, 10], size=(600, 280),
        ylabel=L"y", title=L"n(x, t = " * "$(round(sol.t[end], digits=2)))"), "blob density t=20.pdf")
savefig(heatmap(domain, sol.u[6][:, :, 2], levels=10, aspect_ratio=:equal, xlabel=L"x", ylim=[-10, 10], size=(600, 280),
        ylabel=L"y", title=L"\Omega(x, t = " * "$(round(sol.t[6], digits=2)))", color=:jet), "blob vorticity t=5.pdf")
savefig(heatmap(domain, sol.u[11][:, :, 2], levels=10, aspect_ratio=:equal, xlabel=L"x", ylim=[-10, 10], size=(600, 280),
        ylabel=L"y", title=L"\Omega(x, t = " * "$(round(sol.t[11], digits=2)))", color=:jet), "blob vorticity t=10.pdf")
savefig(heatmap(domain, sol.u[16][:, :, 2], levels=10, aspect_ratio=:equal, xlabel=L"x", ylim=[-10, 10], size=(600, 280),
        ylabel=L"y", title=L"\Omega(x, t = " * "$(round(sol.t[16], digits=2)))", color=:jet), "blob vorticity t=15.pdf")
savefig(heatmap(domain, sol.u[end][:, :, 2], levels=10, aspect_ratio=:equal, xlabel=L"x", ylim=[-10, 10], size=(600, 280),
        ylabel=L"y", title=L"\Omega(x, t = " * "$(round(sol.t[end], digits=2)))", color=:jet), "blob vorticity t=20.pdf")

plot(layout=grid(4, 2), legend=:none, size=(380, 400), dpi=400)
for i in 1:4
    heatmap!(domain.x, domain.y, sol.u[1+5i][:, :, 1], ylim=[-10, 10], aspect_ratio=:equal, subplot=2i - 1)
    heatmap!(domain.x, domain.y, sol.u[1+5i][:, :, 2], ylim=[-10, 10], aspect_ratio=:equal, subplot=2i, color=:jet)
end
savefig("blob evolution.png")

## Recreate Garcia et al. plots Figure 2.
plot(domain.x, sol.u[end][domain.Ny÷2, :, 1], color=:red, xlim=[-5, 25], xlabel=L"x",
    ylabel=L"n(x,t), v_x(x,t)/2", label=L"n(x,t)")
plot!(domain.x, vExB(sol.u[1], domain)[1][domain.Ny÷2, :] / 2, linestyle=:dot, color=:blue, opacity=0.4, label="")
plot!(domain.x, vExB(sol.u[6], domain)[1][domain.Ny÷2, :] / 2, linestyle=:dot, color=:blue, opacity=0.4, label="")
plot!(domain.x, vExB(sol.u[11], domain)[1][domain.Ny÷2, :] / 2, linestyle=:dot, color=:blue, opacity=0.4, label="")
plot!(domain.x, vExB(sol.u[16], domain)[1][domain.Ny÷2, :] / 2, linestyle=:dot, color=:blue, opacity=0.4, label="")
plot!(domain.x, vExB(sol.u[end], domain)[1][domain.Ny÷2, :] / 2, linestyle=:dot, color=:blue, opacity=0.4, label=L"v_x(x,t)/2")
plot!(domain.x, sol.u[1][domain.Ny÷2, :, 1], color=:red, label="")
plot!(domain.x, sol.u[6][domain.Ny÷2, :, 1], color=:red, label="")
plot!(domain.x, sol.u[11][domain.Ny÷2, :, 1], color=:red, label="")
plot!(domain.x, sol.u[16][domain.Ny÷2, :, 1], color=:red, label="")
plot!(domain.x, sol.u[end][domain.Ny÷2, :, 1], color=:red, xlim=[-5, 25], xlabel=L"x",
    ylabel=L"n(x,t), v_x(x,t)/2", label="")
savefig("blob radial variations.pdf")

## Recreate Garcia et al. plots Figure 3.
probe_data = extract_diagnostic(sol.diagnostics[1].data)
t = sol.diagnostics[1].t
positions = sol.diagnostics[1].args[1]
plot(t, probe_data[1, :], xlabel=L"t", ylabel=L"u(x,t)", label=L"x=" * "$(positions[1])")
plot!(t, probe_data[2, :], label=L"x=" * "$(positions[2])")
plot!(t, probe_data[3, :], label=L"x=" * "$(positions[3])")
plot!(t, probe_data[4, :], label=L"x=" * "$(positions[4])")
savefig("blob probe data.pdf")

## Recreate max velocity plot
tends = 2 * logspace(2, -2, 22)
dts = 1 / 2 * tends / 2000
amplitudes = logspace(-2, 5, 22)
max_velocities = similar(amplitudes)
velocities = Vector{Matrix}(undef, length(amplitudes))

parameters = Dict(
    "nu" => 1e-3,
    "kappa" => 1e-3
)

for (i, A) in enumerate(amplitudes)
    # Update initial initial_condition
    u0 = gaussian.(domain.x', domain.y, A=A, B=0, l=1)
    # Update problem 
    prob = SpectralODEProblem(L, N, [u0;;; zero(u0)], domain, [0, tends[i]], p=parameters, dt=dts[i])
    # Reset diagnostics
    diagnostics = [RadialCOMDiagnostic(1), ProgressDiagnostic(100), PlotDensityDiagnostic(1000),]
    prob = SpectralODEProblem(L, N, [u0;;; zero(u0)], domain, [0, tends[i]], p=parameters, dt=dts[i])
    # Reset diagnostics
    diagnostics = [RadialCOMDiagnostic(1), ProgressDiagnostic(100)]#, PlotDensityDiagnostic(1000),]
    # Update output
    output = Output(prob, 21, diagnostics, "output/Garcia 2005 PoP.h5", store_locally=false, simulation_name=string(A))
    # Solve 
    sol = spectral_solve(prob, MSS3(), output, resume=false)
    # Extract velocity
    velocities[i] = sol.simulation["RadialCOMDiagnostic/data"][:, :]
    # Determine max velocity
    max_velocities[i] = maximum(velocities[i][2, :])

    CUDA.pool_status()
end

plot(amplitudes, max_velocities, xaxis=:log, yaxis=:log, aspect_ratio=:auto, marker=:circle, xlabel=L"\Delta n/N", ylabel="max " * L"V", label="")
savefig("figures/blob velocity linear.pdf")

using JLD
jldopen("output/blob velocity linear.jld", "w") do file
    g = create_group(file, "data")
    g["amplitudes"] = amplitudes
    g["max_velocitites"] = max_velocities
end

using SMTPClient
send_mail("Multiple attachments test", attachment="figures/blob velocity linear.pdf")
close(output)

##
fid = h5open("output/Garcia 2005 PoP.h5")
sim = fid[keys(fid)[end-1]]
data = sim["RadialCOMDiagnostic/data"][1, :]

idx = [10, 7, 4]
key_strings = string.(amplitudes)[idx]
for key in key_strings
    #if sum(fid[key]["RadialCOMDiagnostic/data"][2,1:end].<0) != 0
    #    println(minimum(fid[key]["RadialCOMDiagnostic/data"][2,1:end]))
    #end
    plot!(fid[key]["RadialCOMDiagnostic/t"][:], fid[key]["RadialCOMDiagnostic/data"][2, 1:end], aspect_ratio=:auto)
end
display(plot())
display(plot!(legend=false))

sim = fid[keys(fid)[end]]

(sum(sim["fields"][:, :, 1, 1]) - sum(sim["fields"][:, :, 1, 9])) / sum(sim["fields"][:, :, 1, 1])

t = fid[key_strings[2]]["RadialCOMDiagnostic/t"][:]

using JLD
jldopen("output/blob evolution linear.jld", "w") do file
    g = create_group(file, "data")
    g["10"] = fid[key_strings[1]]["RadialCOMDiagnostic/data"][2, 1:end]
    g["1.0"] = fid[key_strings[2]]["RadialCOMDiagnostic/data"][2, 1:end]
    g["0.1"] = fid[key_strings[3]]["RadialCOMDiagnostic/data"][2, 1:end]
    g["t1"] = fid[key_strings[1]]["RadialCOMDiagnostic/t"][:]
    g["t2"] = fid[key_strings[2]]["RadialCOMDiagnostic/t"][:]
    g["t3"] = fid[key_strings[3]]["RadialCOMDiagnostic/t"][:]
end

plot(amplitudes, max_velocities, xaxis=:log, yaxis=:log, marker=:circle, xlabel=L"\Delta n/N", ylabel="max " * L"V", label="")
savefig("blob velocity linear.pdf")

## ------------------------------ Generate blob GIF ----------------------------------------
data = Array.(sol.u)

default(legend=false, ylim=[-10, 10])
anim = @animate for i in 1:201
    i == 2 ? i = 1 : nothing
    heatmap(domain, data[i][:, :, 1], aspect_ratio=:equal, size=(600, 280))
end
#, size=(600, 280)
#, 
gif(anim, "blob.gif", fps=40)