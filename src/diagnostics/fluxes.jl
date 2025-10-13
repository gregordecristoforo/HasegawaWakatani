# -------------------------------------- Fluxes  -------------------------------------------

#Γ_0(t) = 1/(L_xL_y)∫_0^L_x∫_0^L_y nv_x dydx
# Does not take into account anti-aliasing
# TODO implement quadrature as bonus
function radial_flux(u::U, prob::P, t::T; quadrature=nothing) where
{U<:AbstractArray,P<:SpectralODEProblem,T<:Number}

    domain = prob.domain
    n_hat, Ω_hat = eachslice(u, dims=3)
    ϕ_hat = solve_phi(Ω_hat, domain)
    dϕ_hat = diff_y(ϕ_hat, domain)
    vx = zeros(size(domain)) # TODO cache these perhaps?
    n = similar(vx)
    task_vx = Threads.@spawn mul!(vx, get_bwd(prob), dϕ_hat)
    task_n = Threads.@spawn mul!(n, get_bwd(prob), n_hat)
    wait(task_vx)
    wait(task_n)
    @threads for i in eachindex(n)
        @inbounds vx[i] *= n[i]
    end
    return -sum(vx) / (prob.domain.Lx * prob.domain.Ly) # This is the flux time density^^

    # Old code
    #v_x, v_y = vExB(u, prob.domain)
    #sum(u[:,:,1].*v_x)/(prob.domain.Lx*prob.domain.Ly)
end

# TODO move to ext
# CuArray variant
function radial_flux(u::U, prob::P, t::T; quadrature=nothing) where
{U<:CuArray,P<:SpectralODEProblem,T<:Number}
    domain = prob.domain
    n_hat, Ω_hat = eachslice(u, dims=3)
    ϕ_hat = solve_phi(Ω_hat, domain)
    dϕ_hat = diff_y(ϕ_hat, domain)
    vx = zeros(size(domain)) |> domain.MemoryType # TODO cache these perhaps?
    n = similar(vx)
    mul!(vx, get_bwd(prob), dϕ_hat)
    mul!(n, get_bwd(prob), n_hat)
    vx .*= n
    return -sum(vx) / (prob.domain.Lx * prob.domain.Ly) # This is the flux time density^^
end

function RadialFluxDiagnostic(N::Int=10)
    Diagnostic("Radial flux", radial_flux, N, "radial flux", assumes_spectral_field=true)
end