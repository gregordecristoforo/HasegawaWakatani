## Initialize (alt+enter)
include("../../src/HasegawaWakatini.jl")
cd("tests/HW")
domain = Domain(128, 128, 2 * pi / 0.15, 2 * pi / 0.15, anti_aliased=true)
C_values = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
using Statistics
# Open data file
fid = h5open("Hasagawa-Wakatani C weekend scan.h5", "cw")

## Qualitative inspection of temporal resolution

"""
    C_value         0.1         0.2         0.5         1.0         2.0         5.0 (Not enough data)
    fieldStrides    300-600     600-700     800-1000    500         500-600     4000-5000
    dt              0.3-0.6     0.6-0.7     0.8-1       0.5         0.5-0.6     4-5
"""

# Quantitivaly would conclude with around 500 strides

# Quantitive inspection of temporal resolution (less qualitative atleast)

"""
    C_value         0.1         0.2         0.5         1.0         2.0         5.0 (Not enough data) 
    fieldStrides    10-50       10-50       30-100      100-250     80-250      < 2000
    dt              0.01-0.05   0.01-0.05   0.03-0.1    0.1-0.25     0.08-0.25  < 2
"""

# Quantitivaly would conclude 50

# Open solution
S = 1
simulation = fid[keys(fid)[S]]

# Get data
K = read(simulation["Kinetic energy integral/data"])
P = read(simulation["Potential energy integral/data"])
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
heatmap(domain, data[:,:,1, 179], aspect_ratio=:equal)

# Qualitative
plot(P[end-500000:4000:end-240000], marker=:dot)

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

plot(strides, log.(sqrt.((mean_K.-mean_K[1]).^2))/(maximum(mean_K) - minimum(mean_K)))
plot(strides, log.(sqrt.((mean_P.-mean_P[1]).^2))/(maximum(mean_P) - minimum(mean_P)))
plot(strides, log.(sqrt.((mean_Γ.-mean_Γ[1]).^2))/(maximum(mean_Γ) - minimum(mean_Γ)))

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

plot(strides, length.(indicies_K)/length(indicies_K[1]), label=L"K", title="C = $(C_values[S])")
plot!(strides, length.(indicies_P)/length(indicies_P[1]), label=L"P")
plot!(strides, length.(indicies_Γ)/length(indicies_Γ[1]), label=L"\Gamma", xlabel="Stride/sample rate (N)", ylabel="% of peaks detected")
savefig("Number of peaks detected C=$(C_values[S]), Ns=$Ns.pdf")

(length.(indicies_P)/length(indicies_P[1]))[30]

## Get plots

moving_average(vs,n) = [sum(@view vs[i:(i+n-1)])/n for i in 1:(length(vs)-(n-1))]

K_m = moving_average(K[M:end], 10000)

plot(K_m)

mean(K_m)
mean(K_m)

indicies, heights = findmaxima(P[M:M+M÷2])
plotpeaks(P[M:end];peaks=indicies)
#plot(P[M:1000:M+M÷2])
#plotpeaks(P[M:50:M+M÷2];peaks=indicies, prominences=true, widths=true)

length(indicies)

for (i,group) in enumerate(keys(fid))
    simulation = fid[group]
    
    println(C_values[i])

    K = simulation["Kinetic energy integral/data"][:]
    t = simulation["Kinetic energy integral/t"][:]
    P = simulation["Potential energy integral/data"][:]
    flux = simulation["Radial flux/data"][:]

    display(plot(t[1000000:end], K[1000000:end], xlabel="t", ylabel="K", title="C = $(C_values[i])"))
    display(plot(t[1000000:end], P[1000000:end], xlabel="t", ylabel="P", title="C = $(C_values[i])"))
    display(plot(t[1000000:end], K[1000000:end] .+ P[1000000:end], xlabel="t", ylabel="P+K", title="C = $(C_values[i])"))
    display(plot(t[1000000:end], flux[1000000:end], xlabel="t", ylabel=L"\Gamma", title="C = $(C_values[i])"))
end

## Spectral analysis

logn = read(simulation["Log mode diagnstic/data"])
t = read(simulation["Log mode diagnstic/t"])

N = 100000
plot(domain.ky[2:end], logn[:,1,N], xaxis=:log, title="C = 5.0 (t = $(t[N]))", xlabel=L"k_y = k_x", ylabel=L"\log(|n_k|)")
growth = logn[:,1,N] .- logn[:,1,N-1]

#plot(domain.ky[2:end], growth, xaxis=:log, xlabel=L"k_y = k_x", ylabel=L"\gamma", title="t=$(t[N])")

#logmode = output.simulation["Log mode diagnstic/data"][:,:,:]
#for i in 2:10:1000
#    display(plot((logmode[:,1,i] - logmode[:,1,i-1]), title=i))
#end

# ## Make gif
# default(legend=false)
# @gif for i in axes(data, 4)
#     contourf(data[:, :, 1, i])
# end


# Should aim for 100_000-1_000_000 of points for time series!

simulation["t"]
attributes(simulation)

for (key, val) in prob.p
    write_attribute(simulation, key*"3", val)
end
#attributes(simulation)[key] = val

simulation

read(attributes(simulation)["L_x"])
read_attribute(simulation, "C")

## Energy spectra

u_hat = transform(data[:,:,:,end], domain.transform.FT)
function N(u,p,t) u  end
prob = SpectralODEProblem(N, domain, data[:,:,:,end], [1,2])

plot(domain.kx[1:64], radial_energy_spectra(u_hat,prob,0)[1:64], xaxis=:log, yaxis=:log, xlim=[domain.kx[2],domain.kx[64]])
plot(domain.ky, poloidal_energy_spectra(u_hat,prob,0), xaxis=:log, yaxis=:log, xlim=[domain.ky[2],domain.ky[64]])


average_spectra = zero(poloidal_energy_spectra(u_hat,prob,0))
for i in eachindex(axes(data[:,:,:,24:end])[end]) 
    i += 23
    u_hat = transform(data[:,:,:,i], domain.transform.FT)
    average_spectra .+= poloidal_energy_spectra(u_hat,prob,0)
end

plot(domain.ky, average_spectra, xaxis=:log, yaxis=:log, xlim=[domain.ky[2],domain.ky[65]])
plot!(domain.ky, 1000000*domain.ky.^-1.2)
plot(domain.ky, average_spectra.*(domain.ky.^1.36), xaxis=:log, yaxis=:log, xlim=[domain.ky[2],domain.ky[65]])

plot((average_spectra.*(domain.ky.^2.3))[2:end]/1.64e6,xaxis=:log, yaxis=:log)

"""
    C_value         0.1         0.2         0.5         1.0         2.0         5.0 (Not enough data) 
    alpha           1.15        1.26        1.57        1.97        2.54        4.85

"""