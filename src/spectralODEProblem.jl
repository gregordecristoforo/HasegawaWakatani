#include("domain.jl")
using FFTW
export SpectralODEProblem

# TODO add transform method, which is applied in handle_output! to recover function
mutable struct SpectralODEProblem
    L::Function
    N::Function
    domain::Domain
    u0::AbstractArray
    u0_hat::AbstractArray
    tspan::AbstractArray
    p::Dict
    dt::Number
    recover_fields!::Function
    function SpectralODEProblem(N::Function, domain::Domain, u0, tspan; p=Dict(), dt=0.01, 
                                inverse_transformation=identity)
    
        # If no linear operator given, assume there is non
        function L(u, d, p, t)
            zero(u)
        end

        SpectralODEProblem(L, N, domain, u0, tspan, p=p, dt=dt, 
                            inverse_transformation=inverse_transformation)
    end

    function SpectralODEProblem(L::Function, N::Function, domain::Domain, u0, tspan; p=Dict(), 
                                dt=0.01, inverse_transformation=identity)
        u0_hat = transform(u0, domain.transform.FT)
        #u0_hat[:, domain.Nx÷2+1] .= 0
        #u0_hat[domain.Ny÷2+1, :] .= 0

        if length(tspan) != 2
            throw("tspan should have exactly two elements tsart and tend")
        end

        new(L, N, domain, u0, u0_hat, tspan, p, dt, inverse_transformation)
    end
end