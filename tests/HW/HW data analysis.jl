## Initialize (alt+enter)
include(relpath(pwd(), @__DIR__) * "/src/HasegawaWakatini.jl")
cd(relpath(@__DIR__, pwd()))
domain = Domain(128, 128, 2 * pi / 0.15, 2 * pi / 0.15, anti_aliased=true)
C_values = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
using Statistics
# Open data file
fid = h5open("output/Hasegawa-Wakatani C finale march scan.h5", "r")

## Open solution
S = 1
simulation = fid[keys(fid)[S]]

# Get data
K = read(simulation["Kinetic energy integral/data"])
P = read(simulation["Potential energy integral/data"])
U = read(simulation["Enstropy energy integral/data"])
Γ = read(simulation["Radial flux/data"])
data = read(simulation["fields"])
t = read(simulation["Kinetic energy integral/t"])

# Find when "transient period ends"
M = 230000
plot(t[1:2000:M], K[1:2000:M], xlabel=L"t", ylabel=L"P(t)", label="", title="C = $(C_values[S])")

# Visualy confirm with density field
# Get the time
simulation["Kinetic energy integral/t"][M]
simulation["t"][24]
heatmap(domain, data[:, :, 1, 179], aspect_ratio=:equal)



##------------------------------------ Statistics ------------------------------------------

moving_average(vs, n) = [sum(@view vs[i:(i+n-1)]) / n for i in 1:(length(vs)-(n-1))]

K_m = moving_average(K, 5)
plot(K_m)

moving_average(vs, n) = [sum(@view vs[i:(i+n-1)]) / n for i in 1:(length(vs)-(n-1))]

K_m = moving_average(K[M:end], 10000)

indicies, heights = findmaxima(P[M:M+M÷2])
plotpeaks(P[M:end]; peaks=indicies)
#plotpeaks(P[M:50:M+M÷2];peaks=indicies, prominences=true, widths=true)

mean_K = zero(C_values)
mean_P = zero(C_values)
mean_U = zero(C_values)
mean_Γ = zero(C_values)
for (i, group) in enumerate(keys(fid))
    simulation = fid[group]

    K = simulation["Kinetic energy integral/data"][:] / (domain.Lx * domain.Ly)
    P = simulation["Potential energy integral/data"][:] / (domain.Lx * domain.Ly)
    U = simulation["Enstropy energy integral/data"][:] / (domain.Lx * domain.Ly)
    flux = simulation["Radial flux/data"][:] / (domain.Lx * domain.Ly)
    t = simulation["Kinetic energy integral/t"][:]

    plot(t, K, xlabel="t", ylabel="K", title="C = $(C_values[i])", label="")
    display(hline!([mean(K[50:end])], label="⟨K⟩ = $(mean(K[50:end]))"))
    plot(t, P, xlabel="t", ylabel="P", title="C = $(C_values[i])", label="")
    display(hline!([mean(P[50:end])], label="⟨P⟩ = $(mean(P[50:end]))"))
    display(plot(t, K .+ P, xlabel="t", ylabel="P+K", title="C = $(C_values[i])", label=""))
    plot(t, U, xlabel="t", ylabel="U", title="C = $(C_values[i])", label="")
    display(hline!([mean(U[50:end])], label="⟨U⟩ = $(mean(U[50:end]))"))
    plot(t, flux, xlabel="t", ylabel=L"\Gamma", title="C = $(C_values[i])", label="")
    display(hline!([mean(flux[50:end])], label="⟨Γ⟩ = $(mean(flux[50:end]))"))

    mean_K[i] = mean(K[50:end])
    mean_P[i] = mean(P[50:end])
    mean_U[i] = mean(U[50:end])
    mean_Γ[i] = mean(flux[50:end])
end

# TODO make subplot?
plot(C_values, mean_K, xlabel=L"C", ylabel=L"\langle K\rangle", marker=:o, label="")
plot(C_values, mean_P, xlabel=L"C", ylabel=L"\langle P\rangle", marker=:o, label="")
plot(C_values, mean_U, xlabel=L"C", ylabel=L"\langle U\rangle", marker=:o, label="")
plot(C_values, mean_Γ, xlabel=L"C", ylabel=L"\langle Γ\rangle", marker=:o, label="")



##---------------------------------- Spectral analysis -------------------------------------

logn = read(simulation["Log mode diagnostic/data"])
t = read(simulation["Log mode diagnostic/t"])

N = 10
plot(domain.ky[2:end], logn[:, 1, N], xaxis=:log, title="C = 5.0 (t = $(t[N]))", xlabel=L"k_y = k_x", ylabel=L"\log(|n_k|)")
growth = logn[:, 1, N] .- logn[:, 1, N-1]

plot(domain.ky[2:end], growth, xaxis=:log, xlabel=L"k_y = k_x", ylabel=L"\gamma", title="t=$(t[N])")

logmode = output.simulation["Log mode diagnstic/data"][:, :, :]
for i in 2:10:1000
    display(plot((logmode[:, 1, i] - logmode[:, 1, i-1]), title=i))
end

# Should aim for 100_000-1_000_000 of points for time series!

read(attributes(simulation)["L_x"])
read_attribute(simulation, "C")



##--------------------------------- Energy spectra -----------------------------------------

S = 1
simulation = fid[keys(fid)[S]]

u_hat = read(simulation["cache_backup/u"])
n_hat = u_hat[:, :, 1]
omega_hat = u_hat[:, :, 2]

E_k = abs.(n_hat) .^ 2 / ((domain.Nx * domain.Ny)^2)
plot(domain.kx[1:64], vec(sum(E_k, dims=1))[1:64], xaxis=:log, yaxis=:log, xlabel=L"k_x, ky",
    ylabel=L"E", label=L"E(k_x)")
plot!(domain.ky[1:end], vec(sum(E_k, dims=2))[1:end], xaxis=:log, yaxis=:log, label=L"E(k_y)",
    xlim=[0.1, 100], ylim=[1e-20, 20])


#plot(domain.kx[1:64], radial_energy_spectra(u_hat,prob,0)[1:64], xaxis=:log, yaxis=:log, xlim=[domain.kx[2],domain.kx[64]])
#plot(domain.ky, poloidal_energy_spectra(u_hat,prob,0), xaxis=:log, yaxis=:log, xlim=[domain.ky[2],domain.ky[64]])

# average_spectra = zero(poloidal_energy_spectra(u_hat,prob,0))
# for i in eachindex(axes(data[:,:,:,24:end])[end]) 
#     i += 23
#     u_hat = transform(data[:,:,:,i], domain.transform.FT)
#     average_spectra .+= poloidal_energy_spectra(u_hat,prob,0)
# end

# plot(domain.ky, average_spectra, xaxis=:log, yaxis=:log, xlim=[domain.ky[2],domain.ky[65]])
# plot!(domain.ky, 1000000*domain.ky.^-1.2)
# plot(domain.ky, average_spectra.*(domain.ky.^1.36), xaxis=:log, yaxis=:log, xlim=[domain.ky[2],domain.ky[65]])

# plot((average_spectra.*(domain.ky.^2.3))[2:end]/1.64e6,xaxis=:log, yaxis=:log)

"""
    C_value         0.1         0.2         0.5         1.0         2.0         5.0 (Not enough data) 
    alpha           1.15        1.26        1.57        1.97        2.54        4.85

"""



##-------------------------------------- Other ---------------------------------------------

# ## Make gif
default(legend=false)
anim = @animate for i in axes(data, 4)
    heatmap(data[:, :, 1, i])
end
gif(anim, "$(C_values[S]).gif", fps=2)

# Qualitative
plot(P[end-500000:4000:end-240000], marker=:dot)

## Qualitative inspection of temporal resolution

"""
    C_value         0.1         0.2         0.5         1.0         2.0         5.0 (Not enough data)
    fieldStrides    300-600     600-700     800-1000    500         500-600     4000-5000
    dt              0.3-0.6     0.6-0.7     0.8-1       0.5         0.5-0.6     4-5
"""

# Qualitatively would conclude with around 500 strides



# Quantitive
# Difference in mean value
Ns = 10000
strides = [1:Ns...]
mean_K = zeros(Float64, Ns)
mean_P = zeros(Float64, Ns)
mean_Γ = zeros(Float64, Ns)

for i in eachindex(strides)
    mean_K[i] = mean(K[M:strides[i]:end])
    mean_P[i] = mean(P[M:strides[i]:end])
    mean_Γ[i] = mean(Γ[M:strides[i]:end])
end

plot(strides, log.(sqrt.((mean_K .- mean_K[1]) .^ 2)) / (maximum(mean_K) - minimum(mean_K)))
plot(strides, log.(sqrt.((mean_P .- mean_P[1]) .^ 2)) / (maximum(mean_P) - minimum(mean_P)))
plot(strides, log.(sqrt.((mean_Γ .- mean_Γ[1]) .^ 2)) / (maximum(mean_Γ) - minimum(mean_Γ)))

# Quantitive inspection of temporal resolution (less qualitative atleast)

"""
    C_value         0.1         0.2         0.5         1.0         2.0         5.0 (Not enough data) 
    fieldStrides    10-50       10-50       30-100      100-250     80-250      < 2000
    dt              0.01-0.05   0.01-0.05   0.03-0.1    0.1-0.25     0.08-0.25  < 2
"""

# Quantitivaly would conclude 50



# Difference in peaks
using Peaks
Ns = 20000
strides = [1:Ns...]
indicies_K = Vector{AbstractArray}(undef, Ns)
indicies_P = Vector{AbstractArray}(undef, Ns)
indicies_Γ = Vector{AbstractArray}(undef, Ns)
heights_K = Vector{AbstractArray}(undef, Ns)
heights_P = Vector{AbstractArray}(undef, Ns)
heights_Γ = Vector{AbstractArray}(undef, Ns)

for i in eachindex(strides)
    indicies_K[i], heights_K[i] = findmaxima(K[M:strides[i]:end])
    indicies_P[i], heights_P[i] = findmaxima(P[M:strides[i]:end])
    indicies_Γ[i], heights_Γ[i] = findmaxima(Γ[M:strides[i]:end])
end

plot(strides, length.(indicies_K) / length(indicies_K[1]), label=L"K", title="C = $(C_values[S])")
plot!(strides, length.(indicies_P) / length(indicies_P[1]), label=L"P")
plot!(strides, length.(indicies_Γ) / length(indicies_Γ[1]), label=L"\Gamma", xlabel="Stride/sample rate (N)", ylabel="% of peaks detected")
savefig("Number of peaks detected C=$(C_values[S]), Ns=$Ns.pdf")

(length.(indicies_P)/length(indicies_P[1]))[30]





















plot([sum(abs.(irfft(omega_hat[:, i], 128))) for i in 1:128])
plot([sum(abs.(ifft(omega_hat[i, :]))) for i in 1:65])


sum(abs.(n_hat[:, :])) / (domain.Lx * domain.Ly)

# Potential energy 
n = domain.transform.iFT * n_hat
sum(n .^ 2) / (2 * domain.Lx * domain.Ly)
(sum(abs.(n_hat[1:end, :]) .^ 2) - 0.5 * sum(abs.(n_hat[1, :]) .^ 2)) / (domain.Nx * domain.Ny * domain.Lx * domain.Ly)


# Calculate density energy using Parsevals theorem:
E_k = abs.(n_hat) .^ 2
(sum(E_k) - 0.5 * sum(E_k[1, :])) / (domain.Nx * domain.Ny)
1 / 2 * sum(n .^ 2)

#n_hat[y,x]
# Sum across y => radial spectrum
sum(n_hat, dims=1)
# Sum across x => poloidal spectrum
sum(n_hat, dims=2)

# Calculate spectrum
u_hat_y = ifft(n_hat, 1)
plot(sum(abs.(u_hat_y) .^ 2, dims=1)'[1:65], xaxis=:log, yaxis=:log)

u_hat_x = ifft(n_hat, 2)
plot(domain.kx[2:64], sum(abs.(u_hat_x) .^ 2, dims=2)[2:64], xaxis=:log, yaxis=:log)
plot!(domain.ky[2:end], 1e6 * domain.ky[2:end] .^ -3, xaxis=:log, yaxis=:log)
plot!(domain.ky[2:end], 1e6 * domain.ky[2:end] .^ -40, xaxis=:log, yaxis=:log)

# Calculate enstropy energy using Parsevals theorem:
E_k = abs.(omega_hat) .^ 2 #(domain.kx'.^2 .+ domain.ky.^2).*
(sum(E_k) - 0.5 * sum(E_k[1, :])) / (domain.Nx * domain.Ny)
Ω = domain.transform.iFT * omega_hat
U = 1 / 2 * sum(Ω .^ 2)

# Calculate kinetic energy using Parsevals theorem
ϕ =

# kinetic_energy_integral(sol.u[end], prob, 1)
    sum(-domain.SC.Laplacian .* abs.(u_hat[:, :, 2]))

heatmap(abs.(fft(n, 1)))
heatmap(abs.(fft(n, 2)))

plot(sum(abs.(fft(n, 1)), dims=1)', yaxis=:log)
plot(sum(abs.(fft(n, 1)), dims=2), yaxis=:log)
plot(sum(abs.(fft(n, 2)), dims=1)', yaxis=:log)
plot(sum(abs.(fft(n, 2)), dims=2), yaxis=:log)

using BenchmarkTools

E_kx = vec(sum(E_k, dims=1) * 2π / domain.Lx)

surface(log.(fftshift(E_k, 2)), xlabel="x", ylabel="y")

plot(E_kx)


plotlyjsSurface(z=log.(fftshift(E_k, 2)))

fftshift(E_k, domain.Ny)


sum(log.(E_k))