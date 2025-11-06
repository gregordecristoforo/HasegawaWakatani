@testset "output" begin

    Domain(256, 256, Lx=1, Ly=1) # Uses DEFAULT_OPERATORS
    SquareDomain(256, L=1) # Should be same as Domain

    Domain(256, 256, Lx=1, Ly=1, operators=:default)

    # Documentation should explain operators
    Domain(256, 256, Lx=1, Ly=1, operators=:SOL)
    Domain(256, 256, Lx=1, Ly=1, operators=:HW)
    Domain(256, 256, Lx=1, Ly=1, operators=:DW)

    Domain(256, 256, Lx=1, Ly=1, additional_operators=[
            @op ∂x = diff_x,
            @op ∂y = diff_y,
            @op ∂xx = diff_x(order=2)
            @op hyper_laplacian = laplacian(order=3)
        ], aliases=[∂x => :diff_x, ∂y => :diff_y])

    operators = build_operators(domain, kwargs...) # This logic should happen inside SpectralODEProblem

    # Should have operators._domain perhaps, incase user wants to use domain info in rhs

end

SpectralODEProblem(Linear, NonLinear, ic, domain, tspan, p=parameters, dt=1e-3,
    boussinesq=true, additional_operators=[@op ∂x = diff_x, @op ∂y = diff_y,
    @op ∂xx = diff_x(order=2), @op hyper_laplacian = laplacian(order=3)],
    aliases=[∂x => :diff_x, ∂y => :diff_y])

fwd(u)
bwd(u)
∂x(u)
∂y(u)
Δ(u)
spectral_exp(u)
spectral_log(u)
solve_phi(u)
poisson_bracket(u, v)