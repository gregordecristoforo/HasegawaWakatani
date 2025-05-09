using BenchmarkTools
using FFTW
using LinearAlgebra 

function p!(up::DU, u::U, plan::rFFTPlans) where {T, DU<:AbstractArray{T}, U<:AbstractArray{T}}
    up .= zero.(up)
    
    Ny, Nx = size(u)
    Nxl = div(Nx, 2, RoundUp)
    Nxu = div(Nx, 2, RoundDown)
   
    @views @inbounds up[1:Ny, 1:Nxl] .= u[1:Ny, 1:Nxl] # Lower left
    @views @inbounds up[1:Ny, end-Nxu+1:end] .= u[1:Ny, end-Nxu+1:end] # Lower right
    return up
end

# This pads first then mul add
function spectral_conv!(qt::DF, u::F, v::F, U::RF, V::RF, spectral_pad::FP, plans::T) where {
    DF<:AbstractArray,F<:AbstractArray,RF<:AbstractArray,FP<:AbstractArray,T<:TransformPlans}

    mul!(U, plans.iFT, padded ? p!(spectral_pad, u, plans) : u)
    mul!(V, plans.iFT, padded ? p!(spectral_pad, v, plans) : v)    
    @threads for i in eachindex(U)
        U[i] *= V[i]
    end
    mul!(padded ? spectral_pad : qt, plans.FT, U)
    if padded unpad!(qt, spectral_pad, plans) end
end

function test2(qt::DF, u::F, v::F, up::FP, vp::FP, plans::T) where {
    DF<:AbstractArray,F<:AbstractArray,FP<:AbstractArray,T<:TransformPlans}

    p!(up, u, plans)
    p!(vp, v, plans)
    unpad!(qt, spectral_conv(up, vp, plans), plans)
end

function spectral_conv(u_hat, v_hat, plans)
    u = spectral_transform(u_hat, plans.iFT)
    v = spectral_transform(v_hat, plans.iFT)
    spectral_transform(u .* v, plans.FT)
end

function spectral_conv_padded!(qt::DF, u::F, v::F, U::RF, V::RF, spectral_pad::FP, plans::T) where {
    DF<:AbstractArray,F<:AbstractArray,RF<:AbstractArray,FP<:AbstractArray,T<:TransformPlans}
    mul!(U, plans.iFT, p!(spectral_pad, u, plans))
    mul!(V, plans.iFT, p!(spectral_pad, v, plans))    
    @threads for i in eachindex(U)
        U[i] *= V[i]
    end
    #U .*= V
    mul!(spectral_pad, plans.FT, U)
    unpad!(qt, spectral_pad, plans)
end

FFTW.set_num_threads(16)

@btime spectral_conv_padded!(du, u0, v0, U, V, up, QT_plans)
#"73.315"

@btime spectral_conv!(du, u0, v0, U, V, up, QT_plans) 
#"72.671"
@btime test2(dv, u0, v0, up, vp, QT_plans)
#"101.529"
du==dv

function spectral_conv!(qt::DF, u::F, v::F, U::RF, V::RF, up::FP, vp::FP, plans::T) where {
    DF<:AbstractArray,F<:AbstractArray,RF<:AbstractArray,FP<:AbstractArray,T<:TransformPlans}
    # Spawn threads to perform mul! in parallel
    task_U = Threads.@spawn mul!(U, plans.iFT, padded ? p!(up, u, plans) : u)
    task_V = Threads.@spawn mul!(V, plans.iFT, padded ? p!(vp, v, plans) : v)
    # Wait for both tasks to finish
    wait(task_V)
    wait(task_U)

    @threads for i in eachindex(U)
        U[i] *= V[i]
    end
    mul!(padded ? up : qt, plans.FT, U)
    if padded unpad!(qt, up, plans) end
end


@btime spectral_conv!(du, u0, v0, U, V, up, vp, QT_plans) 
#

N = 1024
# Input
u0 = cu(rand(ComplexF64,N ÷ 2 + 1, N))
v0 = cu(rand(ComplexF64,N ÷ 2 + 1, N))

# Pseudo spectral cache
M = 3*N÷2
m = M % 2 == 0 ? M ÷ 2 + 1 : (M - 1) ÷ 2 + 1
up = cu(rand(ComplexF64, m, M))
vp = cu(rand(ComplexF64, m, M))

iFT = plan_irfft(up, M)
FT = plan_rfft(iFT * up)
QT_plans = rFFTPlans(FT, iFT)

U = cu(rand(Float64, M, M))
V = cu(rand(Float64, M, M))

# Output
du = cu(rand(ComplexF64,N ÷ 2 + 1, N))
dv = cu(rand(ComplexF64,N ÷ 2 + 1, N))


using CUDA

function spectral_conv!(qt::DF, u::F, v::F, U::RF, V::RF, up::FP, vp::FP, plans::T) where {
    DF<:CuArray,F<:CuArray,RF<:CuArray,FP<:CuArray,T<:TransformPlans}
    mul!(U, plans.iFT, padded ? p!(up, u, plans) : u)
    mul!(V, plans.iFT, padded ? p!(vp, v, plans) : v)
    @. U = U * V
    mul!(padded ? up : qt, plans.FT, U)
    if padded unpad!(qt, up, plans) end
end

@benchmark CUDA.@sync spectral_conv!(du, u0, v0, U, V, up, vp, QT_plans) 
















































































































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
    #mul!(n)
end

u0 = initial_condition(gaussianWallY, domain)
surface(u0, xlabel="x", ylabel="y")
plot(sum(u0, dims=1)')
plot(sum(u0, dims=2))