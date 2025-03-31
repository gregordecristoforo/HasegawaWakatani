using Plots
include("../../src/domain.jl")
using .Domains
include("../../src/diagnostics.jl")
include("../../src/utilities.jl")

domain = Domain(64, 2)
u0 = initial_condition(gaussian,domain, l=0.08)

N = 1000 
xs = range(-domain.Lx/2, domain.Lx/2-domain.dx, N)
ys = zeros(N)

data = probe(u0, domain, xs, ys, cubic_spline_interpolation)
plot(data)