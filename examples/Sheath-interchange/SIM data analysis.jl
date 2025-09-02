## Run all (alt+enter)
include(relpath(pwd(), @__DIR__) * "/src/HasegawaWakatini.jl")
cd(relpath(@__DIR__, pwd()))

domain = Domain(256, 256, 48, 48, anti_aliased=true)
using Statistics
# Open data file
fid = h5open("output/sheath-interchange long time series.h5", "r")

## Open solution
S = 1
for S in eachindex(keys(fid))
    try
        simulation = fid[keys(fid)[S]]
        try
            data = read(simulation["Density probe/data"])
            t = read(simulation["Density probe/t"])
            display(plot(t, data', xlabel=L"t", ylabel=L"n(0,0)"))
        catch
        end
    catch
    end
end

close(fid)

fid = h5open("output/sheath-interchange april forth.h5", "r")
simulation = fid[keys(fid)[1]]
read_attribute(simulation, "g")

n = read(simulation["Density probe/data"])
n = stack(n)

using Statistics
using StatsPlots
using Distributions

n_n = (n.-mean(n))/std(n)


density(n_n[1:500001], minorticks=true, xlabel=L"(\tilde{n}-\langle\tilde{n}\rangle)\tilde{n}_{rms}",
ylabel=L"P(\tilde{n})", label="", frame=:box,figsize=(3.37, 2.08277),fontfamily= "Computer Modern", 
linewidth=0.75,grid=false)
plot!(Normal(0,1), label=L"N(0,1)", guidefontsize=15, tickfontsize=15, legendfontsize=13, 
markersize=2.25, linewidth=0.75, minorticks=4)
savefig("fluctuation pdf.pdf")

density(n)




keys(fid)
S = 1
simulation = fid[keys(fid)[S]]
data = read(simulation["Density probe/data"])
t = read(simulation["Density probe/t"])
display(plot(t, data', xlabel=L"t", ylabel=L"n(0,0)"))
#delete_object(simulation)

data = simulation["fields"][:, :, :, :]
t = simulation["t"][:]
default(legend=false)
anim = @animate for i in axes(data, 4)
    heatmap(data[:, :, 1, i], aspect_ratio=:equal, xaxis=L"x", yaxis=L"y", title=L"n(t=" * "$(round(t[i], digits=0)))")
end
gif(anim, "delete me.gif", fps=20)

simulation

using Plots
default(legend=true, tickfont=font(12))
x = -6:0.1:6
plots = map(f -> plot(x, f.(x), title=string(f), tickfont=font(12)), [sin, cos, sinh, cosh])
plotgrid = plot(plots..., layout=grid(2,2), link=:both)

#
using JLD
save("density probe.jld", "probe data", n)


# Extract data to do local python analysis
include(relpath(pwd(), @__DIR__) * "/src/HasegawaWakatini.jl")
cd(relpath(@__DIR__, pwd()))
fid = h5open("output/sheath-interchange g=1e-3.h5", "r")

data = fid[keys(fid)[1]]["All probe/data"][:,:,1:1:end]
t = fid[keys(fid)[1]]["All probe/t"][1:10:end]

using JLD
jldopen("output/all probes g=5e-3 dense.jld", "w") do file
    g = create_group(file, "data")
    g["n"] = data[1,1,1:argmax(t)]
    g["Omega"] = data[1,2,1:argmax(t)]
    g["phi"] = data[1,3,1:argmax(t)]
    g["vx"] = data[1,4,1:argmax(t)]
    g["Gamma"] = data[1,5,1:argmax(t)]
    g["t"] = t[1:argmax(t)]
end

fields = fid[keys(fid)[1]]["fields"][:,:,:,:]
heatmap(fields[:,:,1,25], aspect_ratio=:equal)

@views n = data[1,1,:]

plot(n[500:10:end], marker=".")

data = fid[keys(fid)[1]]["Radial flux/data"][:,:,:]
data = fid[keys(fid)[1]]["Enstropy energy integral/data"][:]
data = fid[keys(fid)[1]]["Potential energy integral/data"][:]
data = fid[keys(fid)[1]]["All probe/data"][:,:,1:10:end]

"Radial flux"
"Kinetic energy integral"

#data = fid[keys(fid)[1]]["Radial flux/data"][1:10:5000001]
data = fid[keys(fid)[1]]["Enstropy energy integral/data"][:]

plot(data[end-100:1:end], marker=".")
fid[keys(fid)[1]]

data = fid[keys(fid)[1]]["fields"][:,:,:,:]
heatmap(data[:,:,1,end])#,aspect_ratio=:equal)




include(relpath(pwd(), @__DIR__) * "/src/HasegawaWakatini.jl")
cd(relpath(@__DIR__, pwd()))
fid = h5open("output/debug.h5", "r")
N = read(fid["10 probes too/cache_backup/last_step"])
data = fid["10 probes too/All probe/data"][:,:,1:N÷10]
t = fid["10 probes too/All probe/t"][1:N÷10]

using JLD
jldopen("output/all probes g=1e-2 10 probes.jld", "w") do file
    g = create_group(file, "data")
    g["n"] = data[:,1,1:argmax(t)]
    g["Omega"] = data[:,2,1:argmax(t)]
    g["phi"] = data[:,3,1:argmax(t)]
    g["vx"] = data[:,4,1:argmax(t)]
    g["Gamma"] = data[:,5,1:argmax(t)]
    g["t"] = t[1:argmax(t)]
end



data[1,1,:]

data = sim["fields"][:,:,:,:]

heatmap(data[:,:,1,25])

fid = h5open("output/sheath-interchange g=1e-2.h5", "r")
sim = fid["D_n=0.01, D_Ω=0.01, N=1.0, g=0.01, kappa=0.31622776601683794, sigma_n=0.001, sigma_Ω=0.001"]


## ------------------------------- Gyro Bohm -----------------------------------------------
fid = h5open("output/gyro-bohm=5e-2.h5", "r")
sim = fid["D=0.01, g=0.1, sigma=0.05"]
t = sim["All probe/t"][1:3_079_300]
data = sim["All probe/data"][:,:,1:3_079_300]

sim["All probe/t"][3_079_300]


argmax(t)
heatmap(sim["fields"][:,:,1,argmax(sim["t"][:])], levels=17)#, cmap=:black)

using JLD
jldopen("output/all probes gyro-bohm=5e-2 10 probes.jld", "w") do file
    g = create_group(file, "data")
    g["n"] = data[:,1,1:argmax(t)]
    g["Omega"] = data[:,2,1:argmax(t)]
    g["phi"] = data[:,3,1:argmax(t)]
    g["vx"] = data[:,4,1:argmax(t)]
    g["Gamma"] = data[:,5,1:argmax(t)]
    g["t"] = t[1:argmax(t)]
end