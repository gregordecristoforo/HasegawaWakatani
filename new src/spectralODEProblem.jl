include("domain.jl")
export SpectralODEProblem

mutable struct SpectralODEProblem
    f::Function
    domain::Domain
    u0::Array
    tspan::Array
    #bc::Something
    p::Dict
    dt::Number
    function SpectralODEProblem(f, domain, u0, tspan; p=Dict(), dt=0.01)
        kx, ky = getDomainFrequencies(domain)
        p["kx"], p["ky"] = kx, ky
        p["k2"] = Matrix([kx[i]^2 + ky[j].^2 for i in eachindex(kx), j in eachindex(ky)])
        new(f, domain, u0, tspan, p, dt)
    end
end

function updateDomain!(prob::SpectralODEProblem, domain::Domain)
    prob = SpectralODEProblem(prob.f, domain, prob.u0, prob.tspan, p=prob.p, dt=prob.dt)
end