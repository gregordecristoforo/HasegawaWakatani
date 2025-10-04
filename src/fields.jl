using HasegawaWakatani

"""
    Field(data, domain)
    Field(size, ic, domain) #TODO implement

Bundles an AbstractArray with an AbstractDomain, performing like an Array while preserving 
Domain information. 
"""
mutable struct Field{T,N,A<:AbstractArray{T,N},D<:Domain} <: DenseArray{T,N} # TODO A<:DenseArray? 
    data::A
    domain::D # TODO make AbstractDomain

    # TODO constructor should check that array and domain have same size
end

# --------------------------------- Esential -----------------------------------------------

Base.size(f::Field) = size(f.data)
Base.@propagate_inbounds @inline Base.getindex(f::Field, i...) = getindex(f.data, i...)
Base.@propagate_inbounds @inline Base.setindex!(f::Field, value, i...) = setindex!(f.data, value, i...)

Base.IndexStyle(::Type{<:Field}) = IndexStyle(Array)

Base.strides(f::Field) = strides(f.data)
Base.unsafe_convert(::Type{Ptr{T}}, f::Field{T}) where {T} = Base.unsafe_convert(Ptr{T}, f.data)

function Base.similar(f::Field, ::Type{T}=eltype(f), dims::Dims=size(f)) where {T}
    Field(similar(f.data, T, dims), f.domain)
end
Base.elsize(f::Field) = Base.elsize(f.data)

# --------------------------------- Style --------------------------------------------------

Base.showarg(io::IO, A::Field, toplevel) = print(io, typeof(A), " on a FourierDomain") # TODO fix domain part

# ------------------------------- Broadcasting ---------------------------------------------

Base.BroadcastStyle(::Type{<:Field}) = Broadcast.ArrayStyle{Field}()
function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{Field}}, ::Type{ElType}) where {ElType}
    domain = hasproperty(first(bc.args), :domain) ? first(bc.args).domain : last(bc.args).domain
    Field(similar(Array{ElType}, axes(bc)), domain)
end

## -------------------------------- Test ----------------------------------------------------

# Not sure if needed
#Base.@propagate_inbounds @inline Base.getindex(f::Field, i::Int) = getindex(f.data, i)
#Base.pointer(f::Field) = pointer(f.data)
#Base.parent(f::Field) = parent(f.data)
#Base.iterate(f::Field) = iterate(f.data)
#Base.iterate(f::Field, state) = iterate(f.data, state)
#Base.broadcastable(f::Field) = f

# TODO add CUDA support

domain = Domain(256)
n = Field(ones(256, 256), domain)

# TODO overload spectral operators







































# ------------------------------------------------------------------------------------------

# TODO think of better name as might be confussed with struct field
struct Fields{D<:AbstractDict,N<:Int}
    fields::D
    name_to_index
    comparison_hash::UInt64
    #domain::Domain
    #size::NTuple #{N,D}

    function Fields(pairs::Vararg{Pair{Symbol,D},N}) where {N,D<:AbstractArray}
        fieldsDict = Dict(pairs)
        fields = collect(values(fieldsDict))
        # Check that all fields have the same size
        @assert allequal([size(field) for field in fields]) "All fields needs to have the same size"
        # Check that all fields have the same domain
        @assert allequal([field.domain for field in fields]) "All fields needs to have the same domain"

        comparison_hash = hash(keys(fieldsDict))
        new(fieldsDict)
    end
end

# Implement AbstractArray interface
Base.size(f::Fields) = (length(f.fields),)
Base.getindex(f::Fields, i::Int) = f.fields[i]
Base.setindex!(f::Fields, value, i::Int) = (f.fields[i] = value)

function Base.getproperty(x::Fields, s::Symbol)
    if s in fieldnames(typeof(x))
        getfield(x, s)
    else
        # TODO add better error message
        getfield(x.fields, s)
    end
end

Base.size(A::Fields) = A.size#(data.size...,length(A.fields))

function diff_x(f::Field)
    f.domain.SC.DiffX .* f
end

diff_x(n)
n.domain.SC.DiffX' .* n

function diff_x(f::Field)
    im .* f.domain.kx .* f
end

@time im .* n.domain.kx .* n
@time im .* n.domain.kx .* n.data
@btime diff_x($n);
@btime diff_x2($n);

function diff_x(A::T, d::D) where {T<:AbstractArray,D<:Domain}
    im .* domain.kx .* A
end

@btime diff_x($n.data, $domain);

domain = Domain(256, 256)
domain2 = Domain(10, 10)
domain3 = Domain(256, 256)
n = Field(ones(256, 256), domain)
Ω = Field(ones(256, 256), domain)
T = Field(ones(256, 256), domain)

fields1 = Fields(Dict(:n => n), "Test")
fields2 = Fields(Dict(:n => n), "Test")

p = [Pair(:n, n), Pair(:T, T), Pair(:Ω, Ω)]

a = Dict(:n => n, :T => T)

dΩ = CuArray(2 * ones(256, 256))
dT = CuArray(3 * ones(256, 256))
data = cat(dn, dΩ, dT, dims=3)

fieldsDict = Dict(:n => dn, :Ω => dΩ, :T => dT)

hash(domain)
hash(domain2)
hash(domain3)

domain === domain3