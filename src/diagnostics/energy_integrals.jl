# --------------------------------- Energy integrals ---------------------------------------
# For the most part uses Parsevals theorem, these does not correct for aliasing

# Energy integrals 
# P(t) = ∫dx 1/2n^2
function potential_energy_integral(u::U, prob::P, t::T; quadrature=nothing) where 
    {U<:AbstractArray,P<:SpectralODEProblem,T<:Number}
    if prob.domain.realTransform
        @views (sum(abs.(u[1:end,:,1]).^2) .- 0.5*sum(abs.(u[1,:,1]).^2))/(domain.Nx*domain.Ny*domain.Lx*domain.Ly)
    else
        @views (sum(abs.(u[:,:,1]).^2))/(2.0*domain.Nx*domain.Ny*domain.Lx*domain.Ly)
    end
end

function PotentialEnergyDiagnostic(N::Int = 10)
    Diagnostic("Potential energy integral", potential_energy_integral, N, "potential energy", 
    assumesSpectralField=true)
end

# K(t) = ∫1/2(∇_⟂Φ)^2 = ∫dx1/2 U_E^2
function kinetic_energy_integral(u::U, prob::P, t::T; quadrature=nothing) where 
    {U<:AbstractArray,P<:SpectralODEProblem,T<:Number}
    ϕ = @views solvePhi(u[:,:,2], prob.domain)
    E_kin = -diffusion(abs.(ϕ).^2, prob.domain)

    if prob.domain.realTransform
        @views (sum(E_kin[1:end,:]) .- 0.5*sum(E_kin[1,:]))/(domain.Nx*domain.Ny*domain.Lx*domain.Ly)
    else
        sum(E_kin)/(2.0*domain.Nx*domain.Ny*domain.Lx*domain.Ly)
    end
end

function KineticEnergyDiagnostic(N::Int = 10)
    Diagnostic("Kinetic energy integral", kinetic_energy_integral, N, "kinetic energy", 
    assumesSpectralField=true)
end

# E(t) = P(T) + K(T)
function total_energy_integral(u::U, prob::P, t::T; quadrature=nothing) where 
    {U<:AbstractArray,P<:SpectralODEProblem,T<:Number}
    potential_energy_integral(u, prob, t) .+ kinetic_energy_integral(u, prob, t)
end

function TotalEnergyDiagnostic(N::Int = 10)
    Diagnostic("Total energy integral", total_energy_integral, N, "total energy", 
    assumesSpectralField=true)
end

# U(t) = ∫1/2(∇_⟂^2Φ)^2 = ∫dx1/2 U_E^2
function enstropy_energy_integral(u::U, prob::P, t::T; quadrature=nothing) where 
    {U<:AbstractArray,P<:SpectralODEProblem,T<:Number}
    if prob.domain.realTransform
        @views (sum(abs.(u[1:end,:,2]).^2) .- 0.5*sum(abs.(u[1,:,2]).^2))/(domain.Nx*domain.Ny*domain.Lx*domain.Ly)
    else
        @views (sum(abs.(u[:,:,2]).^2))/(2.0*domain.Nx*domain.Ny*domain.Lx*domain.Ly)
    end
end

function EnstropyEnergyDiagnostic(N::Int = 10)
    Diagnostic("Enstropy energy integral", enstropy_energy_integral, N, "enstropy energy", 
    assumesSpectralField=true)
end

# Γ_c(t) = C∫(n-ϕ)^2
function resistive_dissipation_integral(u::U, prob::P, t::T; quadrature=nothing) where 
    {U<:AbstractArray,P<:SpectralODEProblem,T<:Number}
    h = @views u[:,:,1] .- solvePhi(u[:,:,2], prob.domain)
    if prob.domain.realTransform
        @views prob.p["C"]*(2*sum(abs.(h[1:end,:]).^2) .- sum(abs.(h[1,:]).^2))/(domain.Nx*domain.Ny*domain.Lx*domain.Ly)
    else
        prob.p["C"]*(sum(abs.(h).^2))/(domain.Nx*domain.Ny*domain.Lx*domain.Ly)
    end
end

function ResistiveDissipationDiagnostic(N::Int = 10)
    Diagnostic("Resistive dissipation integral", resistive_dissipation_integral, N, 
    "resistive dissipation energy", assumesSpectralField=true)
end

# D^E_N(t) = ν∫n∇⁶_⟂n
function potential_dissipation_integral(u::U, prob::P, t::T; quadrature=nothing) where 
    {U<:AbstractArray,P<:SpectralODEProblem,T<:Number} 
    D_n_hat = @views -prob.p["D_n"]*hyper_diffusion(abs.(u[:,:,1]).^2, prob.domain)

    if prob.domain.realTransform
        @views (2*sum(D_n_hat[1:end,:]) .- sum(D_n_hat[1,:]))/(domain.Nx*domain.Ny*domain.Lx*domain.Ly)
    else
        sum(D_n_hat)/(domain.Nx*domain.Ny*domain.Lx*domain.Ly)
    end
end

function PotentialDissipationDiagnostic(N::Int = 10)
    Diagnostic("Potential dissipation integral", potential_dissipation_integral, N, 
    "potential dissipation energy", assumesSpectralField=true)
end

# D^E_V(t) = ν∫ϕ∇⁶_⟂Ω = ν∫Ω∇⁴_⟂Ω 
function kinetic_dissipation_integral(u::U, prob::P, t::T; quadrature=nothing) where 
    {U<:AbstractArray,P<:SpectralODEProblem,T<:Number} 
    D_Ω_hat = @views prob.p["D_Ω"]*abs.(diffusion(u[:,:,2], prob.domain)).^2

    if prob.domain.realTransform
        @views (2*sum(D_Ω_hat[1:end,:]) .- sum(D_Ω_hat[1,:]))/(domain.Nx*domain.Ny*domain.Lx*domain.Ly)
    else
        sum(D_Ω_hat)/(domain.Nx*domain.Ny*domain.Lx*domain.Ly)
    end
end

function KineticDissipationDiagnostic(N::Int = 10)
    Diagnostic("Kinetic dissipation integral", kinetic_dissipation_integral, N, 
    "kinetic dissipation energy", assumesSpectralField=true)
end

# D^E(t) = D^E_N(t) + D^E_V(t) 
function viscous_dissipation_integral(u::U, prob::P, t::T; quadrature=nothing) where 
    {U<:AbstractArray,P<:SpectralODEProblem,T<:Number} 
    potential_dissipation_integral(u, prob, t) .+ kinetic_dissipation_integral(u, prob, t)
end

function ViscousDissipationDiagnostic(N::Int = 10)
    Diagnostic("Viscous dissipation integral", viscous_dissipation_integral, N, 
    "viscous dissipation energy", assumesSpectralField=true)
end

# dE/dt(t) = - Γ_c - D^E 
function energy_evolution_integral(u::U, prob::P, t::T; quadrature=nothing) where 
    {U<:AbstractArray,P<:SpectralODEProblem,T<:Number} 
    - radial_flux(u, prob, t) - resistive_dissipation_integral(u, prob, t) .- viscous_dissipation_integral(u, prob, t)
end

function EnergyEvolutionDiagnostic(N::Int = 10)
    Diagnostic("Energy evolution integral", energy_evolution_integral, N, 
    "energy evolution", assumesSpectralField=true)
end

# TODO add enstropy dissipation and evolution, more tricky probably