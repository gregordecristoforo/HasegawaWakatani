using BenchmarkTools
using LinearAlgebra
using FFTW

#u = # Fourier modes
#v = # Fourier modes

struct PseudoSpectralBuffer3{T<:Number, N, A<:AbstractArray{T, N}, P<:AbstractArray{<:Complex, N}}
    U::A  # Physical space of u (may or may not be padded)
    V::A  # Physical space of v (may or may not be padded)
    spectral_pad::P  # Array used to pad Fourier modes, assumed complex
    function PseudoSpectralBuffer3(U::A, V::A, spectral_pad::P) where {T<:Number, N, A<:AbstractArray{T, N}, P<:AbstractArray{<:Complex, N}}
        # Check that U and V have the same size
        if size(U) != size(V)
            throw(ArgumentError("U and V must have the same size"))
        end

        new{T, N, A, P}(U, V, spectral_pad)
    end
end

u = rand(1024, 1024)
v = rand(1024, 1024)
u_hat = rand(ComplexF64, 1024, 1024)

buffer = PseudoSpectralBuffer3(u,v,u_hat)

@unpack U, V, spectral_pad = buffer

using UnPack

funtion conv!(w, u, v, plans, buffer::PseudoSpectralBuffer)
    @unpack up, vp, wp, U, V, W = buffer
    pad!(up, u, plans)
    pad!(vp, v, plans)
    mul!(U, plans.iFT, up)
    mul!(V, plans.iFT, vp)
    @. W = U*V
    mul!(wp, plans.FT, W)
    unpad!(w, wp, plans)
end


funtion spectral_conv!(out::T, u::T, v::T, plans::TransformPlans, buffer::PseudoSpectralBuffer{T}) 
    where {T<:AbstractArray{<:Number}}

    @unpack U, V, spectral_pad = buffer
    pad!(spectral_pad, u, plans)
    mul!(U, plans.iFT, spectral_pad)
    pad!(spectral_pad, v, plans)
    mul!(V, plans.iFT, spectral_pad)
    U *= V 
    mul!(spectral_pad, plans.FT, U)
    unpad!(out, up, spectral_pad)
    out
end




















































#

flux_data = zeros(size(data)[end])
for i in eachindex(flux_data)
    flux_data[i] = radial_flux(data[:,:,:,i], domain)
end

t = fid["2025-02-26T17:33:29.888/t"][:]
plot(t,flux_data.+1e-33)

function v_x_hat(u_hat::AbstractArray{<:Complex}, domain::Domain)
    phi_hat = solvePhi(u_hat[:, :, 2], domain)
    -diffY(phi_hat, domain)
end

function spectral_v_x!(out::AbstractArray{<:Number}, u_hat::AbstractArray{<:Complex}, domain::Domain)
    temp_hat = solvePhi(u_hat[:, :, 2], domain)
    temp_hat .= -diffY(temp_hat, domain)
    mul!(out, domain.transform.iFT, temp_hat)
end

function spectral_radial_flux(u_hat::AbstractArray{<:Complex}, domain::Domain)
    sum(domain.transform.iFT*quadraticTerm(u_hat[:,:,1], v_x_hat(u_hat, domain), domain))/(domain.Lx*domain.Ly)
end

spectral_radial_flux(transform(u,domain.transform.FT), domain)
flux_data[end]

function spectral_flux()
    spectral_v_x!(V, transform(u,domain.transform.FT), domain)
    mul!(n)
end

u0 = initial_condition(gaussianWallY, domain)
surface(u0, xlabel="x", ylabel="y")
plot(sum(u0, dims=1)')
plot(sum(u0, dims=2))