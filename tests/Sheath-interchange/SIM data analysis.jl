## Run all (alt+enter)
include(relpath(pwd(), @__DIR__) * "/src/HasegawaWakatini.jl")
cd(relpath(@__DIR__, pwd()))

domain = Domain(128, 128, 100, 100, anti_aliased=true)
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




using JLD
save("density probe.jld", "probe data", n)