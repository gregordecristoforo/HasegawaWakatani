# -------------------------------------- Fluxes  -------------------------------------------

#Γ_0(t) = 1/(L_xL_y)∫_0^L_x∫_0^L_y nv_x dydx
# TODO implement quadrature and make it more efficient
function radial_flux(u::U, prob::P, t::T; quadrature=nothing) where 
    {U<:AbstractArray,P<:SpectralODEProblem,T<:Number}
    v_x, v_y = vExB(u, prob.domain)
    sum(u[:,:,1].*v_x)/(prob.domain.Lx*prob.domain.Ly)
end

function RadialFluxDiagnostic(N::Int = 10)
    Diagnostic("Radial flux", radial_flux, N, "radial flux")
end