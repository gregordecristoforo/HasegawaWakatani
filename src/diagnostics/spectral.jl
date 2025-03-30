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


function radial_energy_spectra(u::AbstractArray{<:Number}, prob::SpectralODEProblem, t::Number)
    sum(abs.(ifft(u_hat[:,:,1], 1)).^2, dims=1)' #Average?
end

function RadialEnergySpectraDiagnostic(N=100)
    #iFT = ifft() # TODO allocate transform plan
    Diagnostic("Radial energy spectra", radial_energy_spectra, N, "radial energy spectra")
end

function poloidal_energy_spectra(u::AbstractArray{<:Number}, prob::SpectralODEProblem, t::Number)
    sum(abs.(ifft(u_hat[:,:,1] ,2)).^2, dims=2) #Average?
end

function PoloidalEnergySpectraDiagnostic(N=100)
    #iFT = ifft() # TODO allocate transform plan
    Diagnostic("Poloidal energy spectra", poloidal_energy_spectra, N, "poloidal energy spectra")
end