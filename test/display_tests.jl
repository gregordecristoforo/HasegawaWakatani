# ------------------------------------------------------------------------------------------
#                                       Display Test                                        
# ------------------------------------------------------------------------------------------

using HasegawaWakatani
using CUDA
import HasegawaWakatani: build_diagnostic, build_operator

# Minimal construction
domain = Domain(256, 256; MemoryType=CuArray)
ic = initial_condition(isolated_blob, domain) |> HasegawaWakatani.memory_type(domain)
dt = 0.0001

# Emulates SpectralODEProblem
prob = (; domain=domain, operators=(; solve_phi=build_operator(Val(:solve_phi), domain)),
        dt)

# Construct display diagnostic
display_density = build_diagnostic(Val(:plot_density); dt=dt)
# Test density display
display_density(ic, prob, 0.022)

# Construct display diagnostic
display_vorticity = build_diagnostic(Val(:plot_vorticity); dt=dt)
# Test density display
display_vorticity(ic, prob, 0.022)

display_potential = build_diagnostic(Val(:plot_potential); dt=dt)
ic_hat = cat(get_fwd(domain) * ic[:, :, 1], get_fwd(domain) * ic[:, :, 1]; dims=3)
display_potential(ic_hat, prob, 0.0225)

"""
* Test that time = -1 gives different string ending from time != -1
* Test that can construct all display diagnostics and that they work
* Test that the number of significant digits work
"""