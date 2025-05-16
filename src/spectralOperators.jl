module SpectralOperators
using Base.Threads
using CUDA
include("fftutilities.jl")

export SpectralOperatorCache, FFTPlans, diffX, diffY, diffXX, diffYY, diffusion, solvePhi,
    poissonBracket, quadraticTerm, quadraticTerm!

struct SpectralOperatorCache{DX<:AbstractArray,DY<:AbstractArray,DXX<:AbstractArray,
    DYY<:AbstractArray,L<:AbstractArray,SP<:AbstractArray,P<:AbstractArray,DU<:AbstractArray,
    T<:TransformPlans,PHI<:AbstractArray}

    # Spectral coefficents
    DiffX::DX
    DiffY::DY
    DiffXX::DXX
    DiffYY::DYY
    Laplacian::L
    invLaplacian::L
    HyperLaplacian::L

    # Psudo spectral cache
    padded::Bool
    up::SP
    vp::SP
    U::P
    V::P
    qtl::DU
    qtr::DU
    C::Float64                    # Coefficient used for correct anti-aliasing normalization
    QTPlans::T                                               # QT stands for quadratic terms

    # Other cache
    phi::PHI

    function SpectralOperatorCache(kx, ky, Nx, Ny; realTransform=true, anti_aliased=true, use_cuda=false)
        # Spectral coefficents (TODO All have to be CUDA)
        DiffX = transpose(im * kx)
        DiffY = im * ky
        DiffXX = -kx' .^ 2
        DiffYY = -ky .^ 2
        Laplacian = -kx' .^ 2 .- ky .^ 2
        invLaplacian = Laplacian .^ -1
        HyperLaplacian = Laplacian .^ 3
        invLaplacian[1] = 0 # First entry will always be NaN or Inf

        qtl = im * zeros(length(ky), length(kx))
        qtr = zero(qtl)
        phi = zero(qtl)

        if anti_aliased
            N = Nx > 1 ? div(3 * Nx, 2, RoundUp) : 1
            M = Ny > 1 ? div(3 * Ny, 2, RoundUp) : 1
        else
            N, M = Nx, Ny
        end

        # TODO make these CUDA
        if realTransform
            m = M % 2 == 0 ? M ÷ 2 + 1 : (M - 1) ÷ 2 + 1
            spectral_pad = use_cuda ? im * CUDA.zeros(Float64, m, N) : im * zeros(m, N)  #These control precision
            iFT = plan_irfft(im * spectral_pad, M)
            FT = plan_rfft(iFT * spectral_pad)
            QT_plans = rFFTPlans(FT, iFT)
        else
            spectral_pad = use_cuda ? im * CUDA.zeros(Float64, M, N) : im * zeros(M, N)
            FT = plan_fft(spectral_pad)
            iFT = plan_ifft(spectral_pad)
            QT_plans = FFTPlans(FT, iFT)
        end

        up = zero(spectral_pad)
        vp = zero(spectral_pad)

        # Calculate correct conversion coefficent
        C = M * N / (Nx * Ny)
        U = iFT * up
        V = iFT * vp

        if use_cuda # cu to get 32, CuArray to get 64
            DiffX = CuArray(DiffX)
            DiffY = CuArray(DiffY)
            DiffXX = CuArray(DiffXX)
            DiffYY = CuArray(DiffYY)
            Laplacian = CuArray(Laplacian)
            invLaplacian = CuArray(invLaplacian)
            HyperLaplacian = CuArray(HyperLaplacian)
            up = CuArray(up)
            vp = CuArray(vp)
            U = CuArray(U)
            V = CuArray(V)
            qtl = CuArray(qtl)
            qtr = CuArray(qtr)
            phi = CuArray(phi)
            #C = Float32(C) # TODO make less forced
        end

        new{typeof(DiffX),typeof(DiffY),typeof(DiffXX),typeof(DiffYY),typeof(Laplacian),
            typeof(up),typeof(U),typeof(qtr),typeof(QT_plans),typeof(phi)
        }(DiffX, DiffY, DiffXX, DiffYY, Laplacian, invLaplacian, HyperLaplacian,
            anti_aliased, up, vp, U, V, qtl, qtr, C, QT_plans, phi)
    end
end

#------------------------- Quadratic terms interface ---------------------------------------
function quadraticTerm(u::U, v::V, SC::SOC) where {U<:AbstractArray,V<:AbstractArray,
    SOC<:SpectralOperatorCache}
    spectral_conv!(SC.qtl, u, v, SC)
end

# TODO perhaps remove and make alias as it has the same parameters
function quadraticTerm!(out::DF, u::F, v::F, SC::SOC) where {DF<:AbstractArray,
    F<:AbstractArray,SOC<:SpectralOperatorCache}
    spectral_conv!(out, u, v, SC)
end

function spectral_conv!(out::DU, u::U, v::V, SC::SOC) where {DU<:AbstractArray,
    U<:AbstractArray,V<:AbstractArray,SOC<:SpectralOperatorCache}
    plans = SC.QTPlans
    # Spawn threads to perform mul! in parallel
    task_U = Threads.@spawn mul!(SC.U, plans.iFT, SC.padded ? pad!(SC.up, u, plans) : copy(u))
    task_V = Threads.@spawn mul!(SC.V, plans.iFT, SC.padded ? pad!(SC.vp, v, plans) : copy(v))
    # Wait for both tasks to finish
    wait(task_V)
    wait(task_U)

    @threads for i in eachindex(SC.U)
        SC.U[i] *= SC.V[i]
    end
    mul!(SC.padded ? SC.up : out, plans.FT, SC.U)
    SC.padded ? SC.C * unpad!(out, SC.up, plans) : out
end

# Kept for legacy 
function spectral_conv(u_hat, v_hat, plans)
    u = transform(u_hat, plans.iFT)
    v = transform(v_hat, plans.iFT)
    transform(u .* v, plans.FT)
end

# TODO add in a future push
function spectral_conv!(out::DU, u::U, v::V, SC::SOC) where {DU<:CuArray,U<:CuArray,
    V<:CuArray,SOC<:SpectralOperatorCache}

    plans = SC.QTPlans
    mul!(SC.U, plans.iFT, SC.padded ? pad!(SC.up, u, plans) : u)
    mul!(SC.V, plans.iFT, SC.padded ? pad!(SC.vp, v, plans) : v)
    @. SC.U *= SC.V
    mul!(SC.padded ? SC.up : out, plans.FT, SC.U)
    #if padded unpad!(qt, up, plans) end
    SC.padded ? SC.C * unpad!(out, SC.up, plans) : out
end

# Specialized for 2D arrays
# TODO optimize for GPU
function pad!(up::DU, u::U, plan::FFTPlans) where {T,DU<:AbstractArray{T},U<:AbstractArray{T}}
    up .= zero.(up)
    Ny, Nx = size(u)

    Nxl = div(Nx, 2, RoundUp)
    Nxu = div(Nx, 2, RoundDown)
    Nyl = div(Nx, 2, RoundUp)
    Nyu = div(Nx, 2, RoundDown)

    @views @inbounds up[1:Nyl, 1:Nxl] .= u[1:Nyl, 1:Nxl] # Lower left
    @views @inbounds up[1:Nyl, end-Nxu+1:end] .= u[1:Nyl, end-Nxu+1:end] # Lower right
    @views @inbounds up[end-Nyu+1:end, 1:Nxl] .= u[end-Nyu+1:end, 1:Nxl] # Upper left
    @views @inbounds up[end-Nyu+1:end, end-Nxu+1:end] .= u[end-Nyu+1:end, end-Nxu+1:end] # Upper right
    return up
end

# TODO optimize for GPU
function pad!(up::DU, u::U, plan::rFFTPlans) where {T,DU<:AbstractArray{T},U<:AbstractArray{T}}
    up .= zero.(up)
    Ny, Nx = size(u)

    Nxl = div(Nx, 2, RoundUp)
    Nxu = div(Nx, 2, RoundDown)

    @views @inbounds up[1:Ny, 1:Nxl] .= u[1:Ny, 1:Nxl] # Lower left
    @views @inbounds up[1:Ny, end-Nxu+1:end] .= u[1:Ny, end-Nxu+1:end] # Lower right
    return up
end

# TODO optimize for GPU
function unpad!(u::DU, up::U, plan::FFTPlans) where {T,DU<:AbstractArray{T},U<:AbstractArray{T}}
    Ny, Nx = size(u)

    Nyl = div(Nx, 2, RoundUp)
    Nyu = div(Nx, 2, RoundDown)
    Nxl = div(Nx, 2, RoundUp)
    Nxu = div(Nx, 2, RoundDown)

    @views @inbounds u[1:Nyl, 1:Nxl] .= up[1:Nyl, 1:Nxl] # Lower left
    @views @inbounds u[1:Nyl, end-Nxu+1:end] .= up[1:Nyl, end-Nxu+1:end] # Lower right
    @views @inbounds u[end-Nyu+1:end, 1:Nxl] .= up[end-Nyu+1:end, 1:Nxl] # Upper left
    @views @inbounds u[end-Nyu+1:end, end-Nxu+1:end] .= up[end-Nyu+1:end, end-Nxu+1:end] # Upper right
    return u
end

# TODO optimize for GPU
function unpad!(u::DU, up::U, plan::rFFTPlans) where {T,DU<:AbstractArray{T},U<:AbstractArray{T}}
    Ny, Nx = size(u)

    Nxl = div(Nx, 2, RoundUp)
    Nxu = div(Nx, 2, RoundDown)

    @views @inbounds u[1:Ny, 1:Nxl] .= up[1:Ny, 1:Nxl] # Lower left
    @views @inbounds u[1:Ny, end-Nxu+1:end] .= up[1:Ny, end-Nxu+1:end] # Lower right
    return u
end

#-------------------------------- Differentiation ------------------------------------------

function diffX(field::F, SC::SOC) where {F<:AbstractArray,SOC<:SpectralOperatorCache}
    SC.DiffX .* field
end

function diffY(field::F, SC::SOC) where {F<:AbstractArray,SOC<:SpectralOperatorCache}
    SC.DiffY .* field
end

function diffXX(field::F, SC::SOC) where {F<:AbstractArray,SOC<:SpectralOperatorCache}
    SC.DiffXX .* field
end

function diffYY(field::F, SC::SOC) where {F<:AbstractArray,SOC<:SpectralOperatorCache}
    SC.DiffYY .* field
end

function laplacian(field::F, SC::SOC) where {F<:AbstractArray,SOC<:SpectralOperatorCache}
    SC.Laplacian .* field
end

function hyper_diffusion(field::F, SC::SOC) where {F<:AbstractArray,SOC<:SpectralOperatorCache}
    SC.HyperLaplacian .* field
end

function solvePhi(field::F, SC::SOC) where {F<:AbstractArray,SOC<:SpectralOperatorCache}
    SC.phi .= SC.invLaplacian .* field
end

# function poissonBracket(A::U, B::V, SC::SOC) where {U<:AbstractArray,SOC<:SpectralOperatorCache}
#     quadraticTerm(diffX(A, SC), diffY(B, SC), SC) - quadraticTerm(diffY(A, SC), diffX(B, SC), SC)
# end

function poissonBracket(A::U, B::V, SC::SOC) where {U<:AbstractArray,V<:AbstractArray,
    SOC<:SpectralOperatorCache}
    spectral_conv!(SC.qtl, diffX(A, SC), diffY(B, SC), SC) .-= spectral_conv!(SC.qtr, diffY(A, SC), diffX(B, SC), SC)
end

#---------------------------------- Other non-linearities ---------------------------------- 
function spectral_function(f::F, u::U, SC::SOC) where {F<:Function,U<:AbstractArray,SOC<:SpectralOperatorCache}
    plans = SC.QTPlans
    mul!(SC.U, plans.iFT, SC.padded ? pad!(SC.up, u, plans) : u)
    # Assumes function is broadcastable and only 1 argument TODO expand upon this
    SC.V .= f.(SC.C * SC.U)
    mul!(SC.padded ? SC.up : SC.qtl, plans.FT, SC.V)
    SC.padded ? unpad!(SC.qtl, SC.up, plans) / SC.C : SC.qtl
end

end