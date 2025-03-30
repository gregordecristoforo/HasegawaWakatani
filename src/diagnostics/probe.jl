# --------------------------------- Probe --------------------------------------------------

# probe/high time resolution (fields, velocity etc...)

function probe_field(u::AbstractArray, domain::Domain, positions; interpolation=nothing)
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
        U = interpolation((domain.y, domain.x), u)
        for n in eachindex(positions)
            data[n] = U(positions[n][2], positions[n][1])
        end
    end

    return data
end

function probe_density(u::AbstractArray, prob::SpectralODEProblem, t::Number, positions; interpolation=nothing)
    probe_field(u[:, :, 1], prob.domain, positions; interpolation)
end

# "Constructor" for density probe
function ProbeDensityDiagnostic(positions; interpolation=nothing, N=100)::Diagnostic
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

function probe_vorticity(u::AbstractArray, prob::SpectralODEProblem, t::Number, positions; interpolation)
    probe_field(u[:, :, 2], prob.domain, positions; interpolation)
end

# "Constructor" for vorticity probe
function ProbeVorticityDiagnostic(positions; interpolation=nothing, N=100)::Diagnostic
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

# TODO implement remaining probes 

# function ProbePhiDiagnostic(x::Union{AbstractArray,Number}, y::Union{AbstractArray,Number};
#     interpolation=nothing, N=100)::Diagnostic
#     args = (x, y)
#     kwargs = (interpolation = interpolation)
#     return Diagnostic(probe_field, N, "probe", args, kwargs)
# end

# function ProbeVelocityDiagnostic(x::Union{AbstractArray,Number}, y::Union{AbstractArray,Number};
#     interpolation=nothing, N=100)::Diagnostic
#     args = (x, y)
#     kwargs = (interpolation = interpolation)
#     return Diagnostic(probe_field, N, "probe", args, kwargs)
# end