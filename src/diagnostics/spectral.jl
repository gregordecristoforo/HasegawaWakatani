#-------------------------------------- Spectral -------------------------------------------

function get_modes(u::U, prob::P, t::N) where {U<:AbstractArray,P<:SpectralODEProblem,N<:Number}
    return u
end

function GetModeDiagnostic(N::Int=100)
    Diagnostic("Mode diagnostic", get_modes, N, "Modes (Complex)", assumesSpectralField=true)
end

function get_log_modes(u::U, prob::P, t::N; kx::K=:ky) where {U<:AbstractArray,
    P<:SpectralODEProblem,N<:Number,K<:Union{Int,Symbol}}
    if kx == :ky
        if length(size(u)) >= 3
            data = zeros(prob.domain.Nx ÷ 2, last(size(u)))
        else
            data = zeros(prob.domain.Nx ÷ 2)
        end

        for i in 1:prob.domain.Nx÷2
            data[i, :] = log.(abs.(u[i, i, :]))
        end
        return data
    else
        return log.(abs.(u[:, kx, :]))
    end
end

function GetLogModeDiagnostic(N::Int=100, kx::K=:ky) where {K<:Union{Int,Symbol}}
    Diagnostic("Log mode diagnostic", get_log_modes, N, "log(|u_k|)", assumesSpectralField=true, (), (kx=kx,))
end

function plot_frequencies(u::U) where {U<:AbstractArray}
    heatmap(log10.(norm.(u)), title="Frequencies")
end

#--------------------------------- Energy spectra ------------------------------------------

function radial_potential_energy_spectra(u::U, prob::P, t::T) where {U<:AbstractArray,
    P<:SpectralODEProblem,T<:Number}
    sum(abs.(u[:, :, 1]) .^ 2, dims=1)' / (prob.domain.Ny)
end

function RadialPotentialEnergySpectraDiagnostic(N::Int=100)
    Diagnostic("Radial potential energy spectra", radial_potential_energy_spectra, N,
        "radial potential energy spectra", assumesSpectralField=true)
end

function poloidal_potential_energy_spectra(u::U, prob::P, t::T) where {U<:AbstractArray,
    P<:SpectralODEProblem,T<:Number}
    sum(abs.(u[:, :, 1]) .^ 2, dims=2) / (prob.domain.Nx)
end

function PoloidalPotentialEnergySpectraDiagnostic(N::Int=100)
    Diagnostic("Poloidal potential energy spectra", poloidal_potential_energy_spectra, N,
        "poloidal potential energy spectra", assumesSpectralField=true)
end

function radial_kinetic_energy_spectra(u::U, prob::P, t::T) where {U<:AbstractArray,
    P<:SpectralODEProblem,T<:Number}
    sum(abs.(u[:, :, 2]) .^ 2, dims=1)' / (prob.domain.Ny)
end

function RadialKineticEnergySpectraDiagnostic(N::Int=100)
    Diagnostic("Radial kinetic energy spectra", radial_kinetic_energy_spectra, N,
        "radial kinetic energy spectra", assumesSpectralField=true)
end

function poloidal_kinetic_energy_spectra(u::U, prob::P, t::T) where {U<:AbstractArray,
    P<:SpectralODEProblem,T<:Number}
    sum(abs.(u[:, :, 2]) .^ 2, dims=2) / (prob.domain.Nx)
end

function PoloidalKineticEnergySpectraDiagnostic(N::Int=100)
    Diagnostic("Poloidal kinetic energy spectra", poloidal_kinetic_energy_spectra, N,
        "poloidal kinetic energy spectra", assumesSpectralField=true)
end







# radii = -domain.SC.Laplacian
# dk = 0.5*0.15

# radii.÷dk

# radiidx = round.(Int, radii/dk) .+ 1

# bins = zeros(maximum(radiidx))

# for i in eachindex(n_hat)
#     bins[radiidx[i]] += abs(n_hat[i]).^2
# end

# plot(bins .+ 1e-50, xaxis=:log, yaxis=:log)

# bins = 0:dk:maximum(radii)+dk
# energy = zeros(length(bins))

# heatmap((bins[1] .<= radii .<= bins[2]))
# heatmap((bins[2] .<= radii .<= bins[3]))
# heatmap((bins[3] .<= radii .<= bins[4]))
# heatmap((bins[4] .<= radii .<= bins[5]))
# heatmap((bins[5] .<= radii .<= bins[6]))
# heatmap((bins[6] .<= radii .<= bins[7]))
# heatmap((bins[7] .<= radii .<= bins[8]))
# heatmap((bins[8] .<= radii .<= bins[9]))
# heatmap((bins[end-10] .<= radii .<= bins[end]))
# bins[end-3]
# radii[65, 65]
# maximum(radii)

# maximum(sqrt(radii))
# maximum(domain.kx).^2