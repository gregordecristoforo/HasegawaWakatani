## Run all (alt+enter)
include(relpath(pwd(), @__DIR__)*"/src/HasegawaWakatini.jl")
using JLD
using Plots
data = load("reproduced data Garcia 2005 PoP.jld", "data")

domain = Domain(1024, 1024, 50, 50, anti_aliased=false)

# CFL diagnostic

# COM diagnostic
X_COM = zeros(20) 
for i in eachindex(data)
    n = data[i][2:end, 2:end, 1].-1
    X_COM[i] = sum(domain.x[2:end]'.*n)/sum(n)
    #display(contourf(n))
end
V_COM = zeros(20)
V_COM[2:end] = diff(X_COM)

plot(X_COM)
plot(V_COM)

# Probe diagnostic



#initialize_Diagnostic!(diagnostic, u0, prob, t)
#perform_Diagnostic!(diagnostic, U, prob, t)

function perform_Diagnostic!(diagnostic::Diagnostic, U::AbstractArray, prob::SpectralODEProblem, t::Number)
    if isempty(kwargs)
        diagnostic.data[step÷diagnostic.sampleStep] = diagnostic.method(U, prob, t)
    else
        diagnostic.data[step÷diagnostic.sampleStep] = diagnostic.method(U, prob, t, diagnostic.kwargs...)
    end
end


function probe(U, p, t; x, y)
    println("hi")
end

function probe()
    println("testing")
end

function ProbeDiagnostic(x::Union{AbstractArray, Number}, y::Union{AbstractArray, Number}; N=100)
    Diagnostic(probe, 1, "Probe", (x=x, y=y))
end

if empty(NamedTuple())
    println("no")
end


test(1, 10, "", 2; hello=2, mello="")

function test(args...; kwargs...)
    println(args)
    println(kwargs)
end

a = ()
b = (y=2,t=10)

m = [1,2,3]

test(m...)