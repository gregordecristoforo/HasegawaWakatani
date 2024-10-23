include("domain.jl")
using FFTW
export SpectralODEProblem

mutable struct SpectralODEProblem
    f::Function
    domain::Domain
    u0::Array
    u0_hat::Array
    tspan::Array
    #bc::Something
    p::Dict
    dt::Number
    function SpectralODEProblem(f, domain, u0, tspan; p=Dict(), dt=0.01)
        u0_hat = rfft(u0')
        #p["kx"], p["ky"] = domain.kx, domain.ky
        #p["k2"] = Matrix([domain.kx[i]^2 + domain.ky[j] .^ 2 for i in eachindex(domain.kx), j in eachindex(domain.ky)])
        new(f, domain, u0, u0_hat, tspan, p, dt)
    end
end

function updateDomain!(prob::SpectralODEProblem, domain::Domain)
    prob.domain = domain
    kx, ky = getDomainFrequencies(domain)
    prob.p["kx"], prob.p["ky"] = kx, ky
    prob.p["k2"] = Matrix([kx[i]^2 + ky[j] .^ 2 for i in eachindex(kx), j in eachindex(ky)])
end

function updateInitalField!(prob::SpectralODEProblem, initialField::Function)
    prob.u0 = initialField(prob.domain, prob.p)
end

function updateDomain!(prob::SpectralODEProblem, domain::Domain, initialField::Function)
    prob = updateDomain!(prob, domain)
    prob = SpectralODEProblem(prob.f, domain, initialField(domain, prob.p), prob.tspan, p=prob.p, dt=prob.dt)
end

function updateDomain!(prob::SpectralODEProblem, domain::Domain, u0::Array)
    prob = SpectralODEProblem(prob.f, domain, u0, prob.tspan, p=prob.p, dt=prob.dt)
end