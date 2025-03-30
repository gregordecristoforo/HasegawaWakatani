## ----------------------------------------- Parameter study ----------------------------------------------------------------

function parameterStudy(study, values)
    output = similar(values)
    for i in eachindex(values)
        output[i] = study(values[i])
    end
    output
end