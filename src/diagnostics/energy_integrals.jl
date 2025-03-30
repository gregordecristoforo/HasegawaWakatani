# --------------------------------- Functionals --------------------------------------------

#1D profile: n_0(x,t) = 1/L_y∫_0^L_y n(x,y,t)dy
function radial_density_profile(u::AbstractArray{<:Number}, domain::Domain; quadrature=nothing)
    sum(u[:,:,1], dims=1)' ./ domain.Ly
end

function poloidal_density_profile(u::AbstractArray{<:Number}, domain::Domain; quadrature=nothing)
    sum(u[:,:,1], dims=2) ./ domain.Lx
end

function radial_vorticity_profile(u::AbstractArray{<:Number}, domain::Domain; quadrature=nothing)
    sum(u[:,:,2], dims=1)' ./ domain.Ly
end

function poloidal_vorticity_profile(u::AbstractArray{<:Number}, domain::Domain; quadrature=nothing)
    sum(u[:,:,2], dims=2) ./ domain.Lx
end

#Γ_0(x,t) = 1/L_y∫_0^L_y nv_x dy
function radial_flux_profile(u::AbstractArray{<:Number}, domain::Domain; quadrature=nothing)
    v_x, v_y = vExB(u, domain)
    sum(u[:,:,1].*v_x, dims=1)'/domain.Ly
end

#Γ_0(t) = 1/(L_xL_y)∫_0^L_x∫_0^L_y nv_x dydx
# TODO implement quadrature and make it more efficient
function radial_flux(u::AbstractArray{<:Number}, prob::SpectralODEProblem, t::Number; quadrature=nothing)
    v_x, v_y = vExB(u, prob.domain)
    sum(u[:,:,1].*v_x)/(prob.domain.Lx*prob.domain.Ly)
end

function RadialFluxDiagnostic(N = 10)
    Diagnostic("Radial flux", radial_flux, N, "radial flux")
end

# Energy integrals 
# P(t) = ∫dx 1/2n^2
function potential_energy_integral(u::AbstractArray{<:Number}, prob::SpectralODEProblem, 
    t::Number; quadrature=nothing)
    sum(u[:,:,1].^2)/2.0
end

function PotentialEnergyDiagnostic(N = 10)
    Diagnostic("Potential energy integral", potential_energy_integral, N, "potential energy")
end

# K(t) = ∫1/2(∇_⟂Φ)^2 = ∫dx1/2 U_E^2
function kinetic_energy_integral(u::AbstractArray{<:Number}, prob::SpectralODEProblem, 
    t::Number; quadrature=nothing)
    v_x, v_y = vExB(u, domain)
    sum(v_x.^2 .+ v_y.^2)/2.0 
end

function KineticEnergyDiagnostic(N = 10)
    Diagnostic("Kinetic energy integral", kinetic_energy_integral, N, "kinetic energy")
end