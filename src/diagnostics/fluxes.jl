# -------------------------------------- Fluxes  -------------------------------------------
# Does not take into account anti-aliasing

#Γ_0(t) = 1/(L_xL_y)∫_0^L_x∫_0^L_y nv_x dydx
# TODO implement quadrature as bonus
function radial_flux(u::U, prob::P, t::T; quadrature=nothing) where
{U<:AbstractArray,P<:SpectralODEProblem,T<:Number}

    n_hat = u[:, :, 1]
    Ω_hat = @view u[:, :, 2]
    ϕ_hat = solvePhi(Ω_hat, prob.domain)
    dϕ_hat = diffY(ϕ_hat, prob.domain)
    vy = zeros(size(prob.domain.transform.FT)) # TODO cache these perhaps?
    n = similar(vy)
    task_vy = Threads.@spawn mul!(vy, prob.domain.transform.iFT, dϕ_hat)
    task_n = Threads.@spawn mul!(n, prob.domain.transform.iFT, n_hat)
    wait(task_vy)
    wait(task_n)
    @threads for i in eachindex(n)
        @inbounds vy[i] *= n[i]
    end
    return sum(vy) / (prob.domain.Lx * prob.domain.Ly) # This is the flux time density^^

    # Old code
    #v_x, v_y = vExB(u, prob.domain)
    #sum(u[:,:,1].*v_x)/(prob.domain.Lx*prob.domain.Ly)
end

function RadialFluxDiagnostic(N::Int=10)
    Diagnostic("Radial flux", radial_flux, N, "radial flux", assumesSpectralField=true)
end