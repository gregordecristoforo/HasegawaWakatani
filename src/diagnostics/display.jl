#---------------------------- Display diagnostic -------------------------------------------

function plot_density(u::AbstractArray, prob, t::Number)
    digits = ceil(Int, -log10(prob.dt))
    display(heatmap(prob.domain, u[:, :, 1], aspect_ratio=:equal, xlabel="x", ylabel="y",
        title="n(x, t = $(round(t, digits=digits)))"))
end

function PlotDensityDiagnostic(N=1000)
    Diagnostic("Plot density", plot_density, N, "Display density", storesData=false)
end

function plot_vorticity(u::AbstractArray, prob, t::Number)
    digits = ceil(Int, -log10(prob.dt))
    display(heatmap(prob.domain, u[:, :, 2], aspect_ratio=:equal, xlabel="x", ylabel="y",
        title=L"\Omega" * "(x, t = $(round(t, digits=digits)))", color=:jet))
end

function PlotVorticityDiagnostic(N=1000)
    Diagnostic("Plot vorticity", plot_vorticity, N, "Display vorticity", storesData=false)
end

function plot_potential(u::AbstractArray, prob, t::Number)
    d = prob.domain
    phi = d.transform.iFT * solvePhi(u[:, :, 2], d)
    digits = ceil(Int, -log10(prob.dt))
    display(heatmap(prob.domain, phi, aspect_ratio=:equal, xlabel="x", ylabel="y",
        title=L"\Phi" * "(x, t = $(round(t, digits=digits)))"))
end

function PlotPotentialDiagnostic(N=1000)
    Diagnostic("Plot potential", plot_potential, N, "Display potential", assumesSpectralField=true, storesData=false)
end

# TODO is this used?
# ---------------------------------------- Plotting ----------------------------------------

function compareGraphs(x, numerical, analytical; kwargs...)
    plot(x, numerical; label="Numerical", kwargs...)
    plot!(x, analytical; label="Analytical", kwargs...)
end