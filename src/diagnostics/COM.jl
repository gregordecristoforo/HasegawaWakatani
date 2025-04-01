# ------------------------------------ COM -------------------------------------------------

# TODO add argument for using quadratures
function radial_COM(u::U, prob::OP, t::T,p::P) where {U<:AbstractArray,OP<:SpectralODEProblem,
    T<:Number, P<:AbstractArray}
    # 2:end is because the boundaries are periodic and thus should not contribute
    X_COM = sum(prob.domain.x[2:end]' .* u[2:end, 2:end, 1]) / sum(u[2:end, 2:end, 1])

    # Check that do not divide by zero
    if p["previous_time"] == t
        V_COM = 0
    else
        V_COM = (X_COM .- p["previous_position"]) ./ (t .- p["previous_time"])
    end

    # Store 
    p["previous_position"] = X_COM
    p["previous_time"] = t
    [X_COM, V_COM]
end

# "Constructor"
function RadialCOMDiagnostic(N::Int=100)
    #kwargs = (previous_position=0, previous_time=0)
    args = (Dict("previous_position" => 0.0, "previous_time" => 0.0),)
    return Diagnostic("RadialCOMDiagnostic", radial_COM, N, "X_COM, V_COM", args)#, kwargs)
end