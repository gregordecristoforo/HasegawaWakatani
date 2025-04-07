## Run all (alt+enter)
include(relpath(pwd(), @__DIR__)*"/src/HasegawaWakatini.jl")
cd(relpath(@__DIR__, pwd()))

domain = Domain(128, 128, 100, 100, anti_aliased=true)
using Statistics
# Open data file
fid = h5open("output/sheath-interchange april first.h5", "r")

## Open solution
S = 18
simulation = fid[keys(fid)[S]]
data = read(simulation["Density probe/data"])
t = read(simulation["Density probe/t"])
plot(t[1:200], data[1:200], xlabel=L"t", ylabel=L"n(0,0)")
close(fid)

fid = h5open("output/sheath-interchange april second.h5", "r")