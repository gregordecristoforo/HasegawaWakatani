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

# u_hat = transform(sol.u[end], domain.transform.FT)

# n_hat = u_hat[:,:,1]
# omega_hat = u_hat[:,:,2]


# plot([sum(abs.(irfft(n_hat[:,i], 128))) for i in 1:128])
# plot([sum(abs.(ifft(n_hat[i,:]))) for i in 1:65])

# test = fft(n_hat[:,:])

# plot(abs.(test))

# sum(abs.(n_hat[:,:]))/(domain.Lx*domain.Ly)
# sol.diagnostics[4].data[end]


# sum(-domain.SC.Laplacian.*abs.(u_hat[:,:,2]))


# (sum(abs.(n_hat[1:end,:])) - 0.5*sum(abs.(n_hat[1,:])))/(domain.Nx*domain.Ny)
# sol.diagnostics[5].data[end]

# 1/2*sum(sol.u[end][:,:,1].^2)


# test = fft(sol.u[end][:,:,2])

# 1/2*sum(abs.(test).^2)/(128*128) - sol.diagnostics[5].data[end]


# # Calculate density energy using Parsevals theorem:
# E_k = abs.(n_hat).^2 
# (sum(E_k) - 0.5*sum(E_k[1,:]))/(domain.Nx*domain.Ny)
# sol.diagnostics[5].data[end]

# # Calculate kinetic energy using Parsevals theorem:
# E_k = abs.(omega_hat).^2 #(domain.kx'.^2 .+ domain.ky.^2).*
# (sum(E_k) - 0.5*sum(E_k[1,:]))/(domain.Nx*domain.Ny) - sol.diagnostics[4].data[end]

# kinetic_energy_integral(sol.u[end], prob, 1)

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