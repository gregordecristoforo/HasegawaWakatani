#include("domain.jl")
using FFTW
export SpectralODEProblem

mutable struct SpectralODEProblem
    f::Function
    domain::Domain
    u0::AbstractArray
    u0_hat::AbstractArray
    tspan::AbstractArray
    p::Dict
    dt::Number
    function SpectralODEProblem(f, domain, u0, tspan; p=Dict(), dt=0.01)
        u0_hat = rfft(u0)
        if !("nu" in keys(p))
            p["nu"] = 0
        end
        if length(tspan) != 2
            throw("tspan should have exactly two elements tsart and tend")
        end
        new(f, domain, u0, u0_hat, tspan, p, dt)
    end
end

# function updateDomain!(prob::SpectralODEProblem, domain::Domain)
#     prob.domain = domain
#     kx, ky = getDomainFrequencies(domain)
#     prob.p["kx"], prob.p["ky"] = kx, ky
#     prob.p["k2"] = Matrix([kx[i]^2 + ky[j] .^ 2 for i in eachindex(kx), j in eachindex(ky)])
# end

# function updateInitalField!(prob::SpectralODEProblem, initialField::Function)
#     prob.u0 = initialField(prob.domain, prob.p)
# end

# function updateDomain!(prob::SpectralODEProblem, domain::Domain, initialField::Function)
#     prob = updateDomain!(prob, domain)
#     prob = SpectralODEProblem(prob.f, domain, initialField(domain, prob.p), prob.tspan, p=prob.p, dt=prob.dt)
# end

# function updateDomain!(prob::SpectralODEProblem, domain::Domain, u0::AbstractArray)
#     prob = SpectralODEProblem(prob.f, domain, u0, prob.tspan, p=prob.p, dt=prob.dt)
# end