module SpectralOperators

using FFTW, CUDA, Base.Threads, LinearAlgebra, Adapt

include("fftutilities.jl")
export TransformPlans, FFTPlans, rFFTPlans, spectral_transform, spectral_transform!

struct SpectralOperatorCache{DX<:AbstractArray,DY<:AbstractArray,DXX<:AbstractArray,
    DYY<:AbstractArray,L<:AbstractArray,SP<:AbstractArray,P<:AbstractArray,DU<:AbstractArray,
    T<:AbstractFloat,TP<:TransformPlans,PHI<:AbstractArray}

    # Spectral coefficents
    diff_x::DX
    diff_y::DY
    diff_xx::DXX
    diff_yy::DYY
    laplacian::L
    laplacian_inv::L
    hyper_laplacian::L

    # Psudo spectral cache
    padded::Bool
    up::SP
    vp::SP
    U::P
    V::P
    qt_left::DU
    qt_right::DU
    dealiasing_coefficient::T
    QTPlans::TP                                               # QT stands for quadratic terms

    # Other cache
    phi::PHI

    function SpectralOperatorCache(kx, ky, Nx, Ny; use_cuda=true, precision=Float64,
        real_transform=true, dealiased=true)

        # Compute spectral coefficents
        diff_x = transpose(im * kx)
        diff_y = im * ky
        diff_xx = -kx' .^ 2
        diff_yy = -ky .^ 2
        laplacian = -kx' .^ 2 .- ky .^ 2
        laplacian_inv = laplacian .^ -1
        hyper_laplacian = laplacian .^ 3
        # Perhaps a better way exist
        CUDA.@allowscalar laplacian_inv[1] = 0 # First entry will always be NaN or Inf

        # Allocate extra arrays for in-place operations
        qt_left = zeros(complex(precision), length(ky), length(kx))
        use_cuda ? qt_left = adapt(CuArray, qt_left) : nothing
        qt_right = zero(qt_left)
        phi = zero(qt_left)

        # Compute padding length
        if dealiased
            N = Nx > 1 ? div(3 * Nx, 2, RoundUp) : 1
            M = Ny > 1 ? div(3 * Ny, 2, RoundUp) : 1
        else
            N, M = Nx, Ny
        end

        # Determine transform plans for pseudo spectral schemes
        if real_transform
            m = M % 2 == 0 ? M ÷ 2 + 1 : (M - 1) ÷ 2 + 1
            spectral_pad = zeros(complex(precision), m, N)
            use_cuda ? spectral_pad = adapt(CuArray, spectral_pad) : nothing
            iFT = plan_irfft(im * spectral_pad, M)
            FT = plan_rfft(iFT * spectral_pad)
            QT_plans = rFFTPlans(FT, iFT)
        else
            spectral_pad = zeros(complex(precision), M, N)
            use_cuda ? spectral_pad = adapt(CuArray, spectral_pad) : nothing
            FT = plan_fft(spectral_pad)
            iFT = plan_ifft(spectral_pad)
            QT_plans = FFTPlans(FT, iFT)
        end

        # Allocate data for pseudo spectral schemes
        up = zero(spectral_pad)
        vp = zero(spectral_pad)
        U = iFT * up
        V = iFT * vp

        # Calculate correct conversion coefficent
        dealiasing_coefficient = precision(M * N / (Nx * Ny))

        new{typeof(diff_x),typeof(diff_y),typeof(diff_xx),typeof(diff_yy),typeof(laplacian),
            typeof(up),typeof(U),typeof(qt_right),typeof(dealiasing_coefficient),
            typeof(QT_plans),typeof(phi)}(diff_x, diff_y, diff_xx, diff_yy, laplacian,
            laplacian_inv, hyper_laplacian, dealiased, up, vp, U, V, qt_left, qt_right,
            dealiasing_coefficient, QT_plans, phi)
    end
end

# TODO check if operators like laplacians and diff_xx and diff_yy should be Complex 

#------------------------- Quadratic terms interface ---------------------------------------
function quadratic_term(u::U, v::V, SC::SOC) where {U<:AbstractArray,V<:AbstractArray,
    SOC<:SpectralOperatorCache}
    spectral_conv!(SC.qt_left, u, v, SC)
end

# TODO perhaps remove and make alias as it has the same parameters
function quadratic_term!(out::DF, u::F, v::F, SC::SOC) where {DF<:AbstractArray,
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
    SC.padded ? SC.dealiasing_coefficient * unpad!(out, SC.up, plans) : out
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
    SC.padded ? SC.dealiasing_coefficient * unpad!(out, SC.up, plans) : out
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

function diff_x(field::F, SC::SOC) where {F<:AbstractArray,SOC<:SpectralOperatorCache}
    SC.diff_x .* field
end

function diff_y(field::F, SC::SOC) where {F<:AbstractArray,SOC<:SpectralOperatorCache}
    SC.diff_y .* field
end

function diff_xx(field::F, SC::SOC) where {F<:AbstractArray,SOC<:SpectralOperatorCache}
    SC.diff_xx .* field
end

function diff_yy(field::F, SC::SOC) where {F<:AbstractArray,SOC<:SpectralOperatorCache}
    SC.diff_yy .* field
end

# TODO add diff_xn, diff_yn

function laplacian(field::F, SC::SOC) where {F<:AbstractArray,SOC<:SpectralOperatorCache}
    SC.laplacian .* field
end

function hyper_diffusion(field::F, SC::SOC) where {F<:AbstractArray,SOC<:SpectralOperatorCache}
    SC.hyper_laplacian .* field
end

function solve_phi(field::F, SC::SOC) where {F<:AbstractArray,SOC<:SpectralOperatorCache}
    SC.phi .= SC.laplacian_inv .* field
end

# function poisson_bracket(A::U, B::V, SC::SOC) where {U<:AbstractArray,SOC<:SpectralOperatorCache}
#     quadratic_term(diff_x(A, SC), diff_y(B, SC), SC) - quadratic_term(diff_y(A, SC), diff_x(B, SC), SC)
# end

function poisson_bracket(A::U, B::V, SC::SOC) where {U<:AbstractArray,V<:AbstractArray,
    SOC<:SpectralOperatorCache}
    spectral_conv!(SC.qt_left, diff_x(A, SC), diff_y(B, SC), SC) .-= spectral_conv!(SC.qt_right, diff_y(A, SC), diff_x(B, SC), SC) # TODO fix formatting
end

#---------------------------------- Other non-linearities ---------------------------------- 
function spectral_function(f::F, u::U, SC::SOC) where {F<:Function,U<:AbstractArray,SOC<:SpectralOperatorCache}
    plans = SC.QTPlans
    mul!(SC.U, plans.iFT, SC.padded ? pad!(SC.up, u, plans) : u)
    # Assumes function is broadcastable and only 1 argument TODO expand upon this
    SC.V .= f.(SC.dealiasing_coefficient * SC.U)
    mul!(SC.padded ? SC.up : SC.qt_left, plans.FT, SC.V)
    SC.padded ? unpad!(SC.qt_left, SC.up, plans) / SC.dealiasing_coefficient : SC.qt_left
end

end