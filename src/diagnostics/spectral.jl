#-------------------------------------- Spectral -------------------------------------------

function get_modes(u::AbstractArray, prob, t::Number)
    return u
end

function GetModeDiagnostic(N=100)
    Diagnostic("Mode diagnstic", get_modes, N, "Display density", assumesSpectralField=true)
end

function get_log_modes(u::AbstractArray, prob, t::Number; kx=:ky)
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

function GetLogModeDiagnostic(N=100, kx=:ky)
    Diagnostic("Log mode diagnostic", get_log_modes, N, "log(|u_k|)", assumesSpectralField=true, (), (kx=kx,))
end

function plotFrequencies(u)
    heatmap(log10.(norm.(u)), title="Frequencies")
end

#--------------------------------- Energy spectra ------------------------------------------

function radial_potential_energy_spectra(u::U, prob::P, t::T) where {U<:AbstractArray,
    P<:SpectralODEProblem,T<:Number}
    sum(abs.(u[:, :, 1]).^ 2, dims=1)'
end

function RadialPotentialEnergySpectraDiagnostic(N::Int=100)
    Diagnostic("Radial potential energy spectra", radial_potential_energy_spectra, N,
        "radial potential energy spectra", assumesSpectralField=true)
end

function poloidal_potential_energy_spectra(u::U, prob::P, t::T) where {U<:AbstractArray,
    P<:SpectralODEProblem,T<:Number}
    sum(abs.(u[:, :, 1]).^ 2, dims=2)
end

function PoloidalPotentialEnergySpectraDiagnostic(N::Int=100)
    Diagnostic("Poloidal potential energy spectra", poloidal_potential_energy_spectra, N,
        "poloidal potential energy spectra", assumesSpectralField=true)
end

function radial_kinetic_energy_spectra(u::U, prob::P, t::T) where {U<:AbstractArray,
    P<:SpectralODEProblem,T<:Number}
    sum(abs.(u[:, :, 2]) .^ 2, dims=1)'
end

function RadialKineticEnergySpectraDiagnostic(N::Int=100)
    Diagnostic("Radial kinetic energy spectra", radial_kinetic_energy_spectra, N,
        "radial kinetic energy spectra", assumesSpectralField=true)
end

function poloidal_kinetic_energy_spectra(u::U, prob::P, t::T) where {U<:AbstractArray,
    P<:SpectralODEProblem,T<:Number}
    sum(abs.(u[:, :, 2]).^ 2, dims=2)
end

function PoloidalKineticEnergySpectraDiagnostic(N::Int=100)
    Diagnostic("Poloidal kinetic energy spectra", poloidal_kinetic_energy_spectra, N,
        "poloidal kinetic energy spectra", assumesSpectralField=true)
end


# phi_hat = solvePhi(u_hat[:,:,2], domain)

# contourf(domain.transform.iFT*-diffY(phi_hat, domain))










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