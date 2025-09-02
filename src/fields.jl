
struct Field{D<:AbstractArray}
    data::D
    domain::String
end

Base.size(A::Field) = A.data.size
Base.getindex(A, i::Int) = getindex(A.data, i)

n[]
n = Field([], "n")

# TODO think of better name as might be confussed with struct field
struct Fields{D<:AbstractDict,N<:Int}
    fields::D
    domain::String
    size::NTuple #{N,D}

    function Fields(fields)
        # Check that all fields have the same size


    end
end

fields1 = Fields(Dict(:n => n), "Test")
fields2 = Fields(Dict(:n => n), "Test")

function Base.getproperty(x::Fields, s::Symbol)
    if s in fieldnames(typeof(x))
        getfield(x, s)
    else
        # TODO add better error message
        x.fields[s]
    end
end

Base.size(A::Fields) = A.size#(data.size...,length(A.fields))

fields.n
fields.fields
fields.domain

d = Dict(:n => n)

using CUDA
dn = CuArray(ones(256, 256))
dΩ = CuArray(2 * ones(256, 256))
dT = CuArray(3 * ones(256, 256))
data = cat(dn, dΩ, dT, dims=3)

fieldsDict = Dict(:n => dn, :Ω => dΩ, :T => dT)

length(fieldsDict)
size(fields)

dump(dn)

A = ones(256, 256, 3)

fieldnames(CuArray)
fieldnames(Array)
A.ref
A.size

dump(Array)

@edit Base.size(A)