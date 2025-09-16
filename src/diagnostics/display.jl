#---------------------------- Display diagnostic -------------------------------------------

function plot_density(u::U, prob::P, t::T) where {U<:AbstractArray,P<:SpectralODEProblem,T<:Number}
    digits = ceil(Int, -log10(prob.dt))
    display(heatmap(prob.domain, Array(u[:, :, 1]), aspect_ratio=:equal, xlabel="x", ylabel="y",
        title="n(x, t = $(round(t, digits=digits)))"))
end

function PlotDensityDiagnostic(N::Int=1000)
    Diagnostic("Plot density", plot_density, N, "Display density", stores_data=false)
end

function plot_vorticity(u::U, prob::P, t::T) where {U<:AbstractArray,P<:SpectralODEProblem,T<:Number}
    digits = ceil(Int, -log10(prob.dt))
    display(heatmap(prob.domain, Array(u[:, :, 2]), aspect_ratio=:equal, xlabel="x", ylabel="y",
        title=L"\Omega" * "(x, t = $(round(t, digits=digits)))", color=:jet))
end

function PlotVorticityDiagnostic(N::Int=1000)
    Diagnostic("Plot vorticity", plot_vorticity, N, "Display vorticity", stores_data=false)
end

function plot_potential(u::U, prob::P, t::T) where {U<:AbstractArray,P<:SpectralODEProblem,T<:Number}
    d = prob.domain
    phi = d.transform.iFT * solve_phi(u[:, :, 2], d)
    digits = ceil(Int, -log10(prob.dt))
    display(heatmap(prob.domain, Array(phi), aspect_ratio=:equal, xlabel="x", ylabel="y",
        title=L"\Phi" * "(x, t = $(round(t, digits=digits)))"))
end

function PlotPotentialDiagnostic(N::Int=1000)
    Diagnostic("Plot potential", plot_potential, N, "Display potential", assumes_spectral_field=true, stores_data=false)
end