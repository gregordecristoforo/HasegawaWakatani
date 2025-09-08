# ----------------------------------- Profiles ---------------------------------------------

#1D profile: n_0(x,t) = 1/L_y∫_0^L_y n(x,y,t)dy
function radial_density_profile(u::U, prob::P, t::T; quadrature=nothing) where {U<:AbstractArray,
    P<:SpectralODEProblem,T<:Number}
    sum(u[:, :, 1], dims=1)' ./ prob.domain.Ly
end

function poloidal_density_profile(u::U, prob::P, t::T; quadrature=nothing) where {U<:AbstractArray,
    P<:SpectralODEProblem,T<:Number}
    sum(u[:, :, 1], dims=2) ./ prob.domain.Lx
end

function radial_vorticity_profile(u::U, prob::P, t::T; quadrature=nothing) where {U<:AbstractArray,
    P<:SpectralODEProblem,T<:Number}
    sum(u[:, :, 2], dims=1)' ./ prob.domain.Ly
end

function poloidal_vorticity_profile(u::U, prob::P, t::T; quadrature=nothing) where {U<:AbstractArray,
    P<:SpectralODEProblem,T<:Number}
    sum(u[:, :, 2], dims=2) ./ prob.domain.Lx
end

#Γ_0(x,t) = 1/L_y∫_0^L_y nv_x dy
function radial_flux_profile(u::U, prob::P, t::T; quadrature=nothing) where {U<:AbstractArray,
    P<:SpectralODEProblem,T<:Number}
    v_x, v_y = vExB(u, prob.domain)
    sum(u[:, :, 1] .* v_x, dims=1)' / prob.domain.Ly
end