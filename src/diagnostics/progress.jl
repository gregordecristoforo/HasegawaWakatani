# --------------------------- Progress diagnostic ------------------------------------------

function progress(u::U, prob::P, t::T) where {U<:AbstractArray,P<:SpectralODEProblem,T<:Number}
    procentage = (t - first(prob.tspan)) / (last(prob.tspan) - first(prob.tspan)) * 100
    # Determine smallest "digits" to display unique procentage each time
    digits = ceil(Int, -log10(prob.dt / (last(prob.tspan) - first(prob.tspan)))) - 2
    println("$(round(procentage, digits=digits))% done")
end

function ProgressDiagnostic(N::Int=100)
    Diagnostic("Progress", progress, N, "progress", assumesSpectralField=true, storesData=false)
end