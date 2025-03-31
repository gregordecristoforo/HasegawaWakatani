# --------------------------------- Energy integrals ---------------------------------------

# Energy integrals 
# P(t) = ∫dx 1/2n^2
function potential_energy_integral(u::AbstractArray{<:Number}, prob::SpectralODEProblem, 
    t::Number; quadrature=nothing)
    sum(u[:,:,1].^2)/2.0
end

function PotentialEnergyDiagnostic(N = 10)
    Diagnostic("Potential energy integral", potential_energy_integral, N, "potential energy")
end

# TODO Improve how it is calculated
# K(t) = ∫1/2(∇_⟂Φ)^2 = ∫dx1/2 U_E^2
function kinetic_energy_integral(u::AbstractArray{<:Number}, prob::SpectralODEProblem, 
    t::Number; quadrature=nothing)
    v_x, v_y = vExB(u, prob.domain)
    sum(v_x.^2 .+ v_y.^2)/2.0 
end

function KineticEnergyDiagnostic(N = 10)
    Diagnostic("Kinetic energy integral", kinetic_energy_integral, N, "kinetic energy")
end

# U(t) = ∫1/2(∇_⟂^2Φ)^2 = ∫dx1/2 U_E^2
function enstropy_energy_integral(u::AbstractArray{<:Number}, prob::SpectralODEProblem, 
    t::Number; quadrature=nothing)
    sum(u[:,:,2].^2)/2.0 
end

function EnstropyEnergyDiagnostic(N = 10)
    Diagnostic("Enstropy energy integral", enstropy_energy_integral, N, "enstropy energy")
end