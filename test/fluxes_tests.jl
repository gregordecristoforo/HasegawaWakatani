# ------------------------------------------------------------------------------------------
#                                        Flux Tests                                         
# ------------------------------------------------------------------------------------------

using HasegawaWakatani
using CUDA
import HasegawaWakatani: build_diagnostic, build_operator

# Minimal construction
domain = Domain(256, 256; MemoryType=CuArray)
ic = initial_condition(random_crossphased, domain) |> HasegawaWakatani.memory_type(domain)
dt = 0.0001

# Emulates SpectralODEProblem
prob = (; domain=domain,
        operators=(; diff_x=build_operator(Val(:diff_x), domain),
                   diff_y=build_operator(Val(:diff_y), domain),
                   solve_phi=build_operator(Val(:solve_phi), domain)),
        dt=dt)

radial_flux = build_diagnostic(Val(:radial_flux))

ic_hat = cat(get_fwd(domain) * ic[:, :, 1], get_fwd(domain) * ic[:, :, 2]; dims=3)

radial_flux(ic_hat, prob, 0.0)

poloidal_flux = build_diagnostic(Val(:poloidal_flux))
poloidal_flux(ic_hat, prob, 0.0)

"""
* Test that time = -1 gives different string ending from time != -1
* Test that can construct all display diagnostics and that they work
* Test that the number of significant digits work
"""