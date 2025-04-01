## ----------------------------------------- Parameter study ----------------------------------------------------------------

function parameterStudy(study::F, values<:V) where {F<:Function,V<:AbstractArray}
    output = similar(values)
    for i in eachindex(values)
        output[i] = study(values[i])
    end
    output
end