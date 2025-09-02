# --------------------------------- Probe --------------------------------------------------

# probe/high time resolution (fields, velocity etc...)

function probe_field(u::U, domain::D, positions::P; interpolation::I=nothing) where {
    U<:AbstractArray,D<:Domain,P<:Union{AbstractArray,Tuple,Number},I<:Union{Nothing,Function}}
    # Check if the user sent in tuple of points or single point
    if isa(positions, Tuple) && isa(positions[1], Number)
        positions = [positions]
    end

    # Initilize vectors
    data = zeros(length(positions))

    if isnothing(interpolation)
        for n in eachindex(positions)
            i = argmin(abs.(domain.x .- positions[n][1]))
            j = argmin(abs.(domain.y .- positions[n][2]))
            data[n] = u[j, i]
        end
    else
        # Only want to do this once
        u_interpolated = interpolation((domain.y, domain.x), u)
        for n in eachindex(positions)
            data[n] = u_interpolated(positions[n][2], positions[n][1])
        end
    end

    # Return either the one point, or the array
    length(data) == 1 ? data[1] : data
end

function probe_density(u::U, prob::SOP, t::N, positions::P; interpolation::I=nothing) where {
    U<:AbstractArray,SOP<:SpectralODEProblem,N<:Number,P<:Union{AbstractArray,Tuple,Number},
    I<:Union{Nothing,Function}}
    probe_field(u[:, :, 1], prob.domain, positions; interpolation)
end

# "Constructor" for density probe
function ProbeDensityDiagnostic(positions::P; interpolation::I=nothing, N::Int=100) where {
    P<:Union{AbstractArray,Tuple,Number},I<:Union{Nothing,Function}}
    # Check if the user sent in tuple of points or single point
    if isa(positions, Tuple) && isa(positions[1], Number)
        positions = [positions]
    end

    #Create the diagnostic label
    label = ["Probe " * string(position) for position in positions]

    args = (positions,)
    kwargs = (interpolation=interpolation,)

    return Diagnostic("Density probe", probe_density, N, label, args, kwargs)
end

function probe_vorticity(u::U, prob::SOP, t::N, positions::P; interpolation::I=nothing) where {
    U<:AbstractArray,SOP<:SpectralODEProblem,N<:Number,P<:Union{AbstractArray,Tuple,Number},
    I<:Union{Nothing,Function}}
    probe_field(u[:, :, 2], prob.domain, positions; interpolation)
end

# "Constructor" for vorticity probe
function ProbeVorticityDiagnostic(positions::P; interpolation::I=nothing, N::Int=100) where {
    P<:Union{AbstractArray,Tuple,Number},I<:Union{Nothing,Function}}
    # Check if the user sent in tuple of points or single point
    if isa(positions, Tuple) && isa(positions[1], Number)
        positions = [positions]
    end

    #Create the diagnostic label
    label = ["Probe " * string(position) for position in positions]

    args = (positions,)
    kwargs = (interpolation=interpolation,)

    return Diagnostic("Vorticity probe", probe_vorticity, N, label, args, kwargs)
end

function probe_potential(u::U, prob::SOP, t::N, positions::P; interpolation::I=nothing) where {
    U<:AbstractArray,SOP<:SpectralODEProblem,N<:Number,P<:Union{AbstractArray,Tuple,Number},
    I<:Union{Nothing,Function}}
    ϕ_hat = @views solvePhi(u[:, :, 2], prob.domain)
    ϕ = prob.domain.transform.iFT * ϕ_hat
    probe_field(ϕ, prob.domain, positions; interpolation)
end

function ProbePotentialDiagnostic(positions::P; interpolation::I=nothing, N::Int=100) where {
    P<:Union{AbstractArray,Tuple,Number},I<:Union{Nothing,Function}}
    # Check if the user sent in tuple of points or single point
    if isa(positions, Tuple) && isa(positions[1], Number)
        positions = [positions]
    end

    #Create the diagnostic label
    label = ["Probe " * string(position) for position in positions]

    args = (positions,)
    kwargs = (interpolation=interpolation,)

    return Diagnostic("Phi probe", probe_potential, N, label, args, kwargs, assumesSpectralField=true)
end

function probe_radial_velocity(u::U, prob::SOP, t::N, positions::P; interpolation::I=nothing) where {
    U<:AbstractArray,SOP<:SpectralODEProblem,N<:Number,P<:Union{AbstractArray,Tuple,Number},
    I<:Union{Nothing,Function}}
    ϕ_hat = @views solvePhi(u[:, :, 2], prob.domain)
    v_x_hat = -diffY(ϕ_hat, prob.domain)
    v_x = prob.domain.transform.iFT * v_x_hat
    probe_field(v_x, prob.domain, positions; interpolation)
end

function ProbeRadialVelocityDiagnostic(positions::P; interpolation::I=nothing, N::Int=100) where {
    P<:Union{AbstractArray,Tuple,Number},I<:Union{Nothing,Function}}
    # Check if the user sent in tuple of points or single point
    if isa(positions, Tuple) && isa(positions[1], Number)
        positions = [positions]
    end

    #Create the diagnostic label
    label = ["Probe " * string(position) for position in positions]

    args = (positions,)
    kwargs = (interpolation=interpolation,)

    return Diagnostic("Radial velocity probe", probe_radial_velocity, N, label, args, kwargs, assumesSpectralField=true)
end

function probe_all(u::U, prob::SOP, t::N, positions::P; interpolation::I=nothing) where {
    U<:AbstractArray,SOP<:SpectralODEProblem,N<:Number,P<:Union{AbstractArray,Tuple,Number},
    I<:Union{Nothing,Function}}

    # Calculate spectral fields
    ϕ_hat = @views solvePhi(u[:, :, 2], prob.domain)
    v_x_hat = -diffY(ϕ_hat, prob.domain)

    # Cache for transformation
    cache = zeros(size(prob.domain.transform.FT))

    # Transform to physical space and probe fields
    n = mul!(cache, prob.domain.transform.iFT, u[:, :, 1])
    n_p = probe_field(n, prob.domain, positions; interpolation)
    Ω = mul!(cache, prob.domain.transform.iFT, u[:, :, 2])
    Ω_p = probe_field(Ω, prob.domain, positions; interpolation)
    ϕ = mul!(cache, prob.domain.transform.iFT, ϕ_hat)
    ϕ_p = probe_field(ϕ, prob.domain, positions; interpolation)
    v_x = mul!(cache, prob.domain.transform.iFT, v_x_hat)
    v_x_p = probe_field(v_x, prob.domain, positions; interpolation)

    #Combine fields for output (The last field is the flux Γ=nvₓ)
    [n_p;; Ω_p;; ϕ_p;; v_x_p;; n_p .* v_x_p]
end

function ProbeAllDiagnostic(positions::P; interpolation::I=nothing, N::Int=100) where {
    P<:Union{AbstractArray,Tuple,Number},I<:Union{Nothing,Function}}
    # Check if the user sent in tuple of points or single point
    if isa(positions, Tuple) && isa(positions[1], Number)
        positions = [positions]
    end

    #Create the diagnostic label
    label = ["Probe " * string(position) for position in positions]

    args = (positions,)
    kwargs = (interpolation=interpolation,)

    return Diagnostic("All probe", probe_all, N, label, args, kwargs, assumesSpectralField=true)
end

export ProbeDensityDiagnostic, ProbePotentialDiagnostic, ProbeVorticityDiagnostic, ProbeRadialVelocityDiagnostic,
    ProbeAllDiagnostic