module SpectralOperators
include("fftutilities.jl")

export SpectralOperatorCache, FFTPlans, diffX, diffY, diffXX, diffYY, diffusion, solvePhi,
    poissonBracket, quadraticTerm

struct SpectralOperatorCache
    # Spectral coefficents
    DiffX::AbstractArray
    DiffY::AbstractArray
    DiffXX::AbstractArray
    DiffYY::AbstractArray
    Laplacian::AbstractArray
    invLaplacian::AbstractArray
    # QT stands for quadratic terms
    QT::AbstractArray
    up::AbstractArray
    vp::AbstractArray
    QTp::AbstractArray
    U::AbstractArray
    V::AbstractArray
    QTPlans::TransformPlans
    C::Number
    function SpectralOperatorCache(kx, ky, Nx, Ny; realTransform=true, anti_aliased=true)
        # Spectral coefficents
        DiffX = im * kx'
        DiffY = im * ky
        DiffXX = -kx' .^ 2
        DiffYY = -ky .^ 2
        Laplacian = -kx' .^ 2 .- ky .^ 2
        invLaplacian = Laplacian .^ -1
        invLaplacian[1] = 0 # First entry will always be NaN or Inf
        QT = im * zeros(length(ky), length(kx))

        if anti_aliased
            N = Nx > 1 ? ceil(Int, 3 * Nx / 2) : 1
            M = Ny > 1 ? ceil(Int, 3 * Ny / 2) : 1
        else
            N, M = Nx, Ny
        end

        if realTransform
            m = M % 2 == 0 ? M ÷ 2 + 1 : (M - 1) ÷ 2 + 1
            QTp = im * zeros(m, N)
            iFT = plan_irfft(im * QTp, M)
            FT = plan_rfft(iFT * QTp)
            QT_plans = rFFTPlans(FT, iFT)
        else
            QTp = im * zeros(M, N)
            FT = plan_fft(QTp)
            iFT = plan_ifft(QTp)
            QT_plans = FFTPlans(FT, iFT)
        end

        up = im * zero(QTp)
        vp = im * zero(QTp)

        # Count number of dimensions that has more than one dimension along an axis
        C = M * N / (Nx * Ny)
        U = similar(QTp)
        V = similar(QTp)
        new(DiffX, DiffY, DiffXX, DiffYY, Laplacian, invLaplacian, QT, up, vp, QTp, U, V, QT_plans, C)
    end
end

#------------------------- Quadratic terms interface ---------------------------------------
function quadraticTerm(u, v, SC::SpectralOperatorCache)
    if length(u) != length(SC.up)
        pad!(SC.up, u, SC.QTPlans)
        pad!(SC.vp, v, SC.QTPlans)
        unpad!(SC.QT, spectral_conv(SC.up, SC.vp, SC.QTPlans), SC.QTPlans)
        SC.C * SC.QT
    else
        spectral_conv(u, v, SC.QTPlans)
    end
end

function quadraticTerm!(dqt, u, v, SC::SpectralOperatorCache)
    if length(u) != length(SC.up)
        pad!(SC.up, u, SC.QTPlans)
        pad!(SC.vp, v, SC.QTPlans)
        unpad!(SC.QT, spectral_conv(SC.up, SC.vp, SC.QTPlans), SC.QTPlans)
        SC.C * SC.QT
    else
        spectral_conv(u, v, SC.QTPlans)
    end
end

function spectral_conv(u_hat, v_hat, plans)
    u = transform(u_hat, plans.iFT)
    v = transform(v_hat, plans.iFT)
    transform(u .* v, plans.FT)
end

function spectral_conv!(SC, u_hat, v_hat)
    mul!(SC.U, SC.plans.iFT, u_hat)
    mul!(SC.V, SC.plans.iFT, v_hat)
    SC.QTp .= SC.U.*SC.V
    mul!(SC.qtp, SC.plans.FT, sc.QTp)
end

# Specialized for 2D arrays
function pad!(up, u, plan::FFTPlans)
    Ny, Nx = size(u)

    Nxl = ceil(Int, Nx / 2)
    Nxu = floor(Int, Nx / 2)
    Nyl = ceil(Int, Ny / 2)
    Nyu = floor(Int, Ny / 2)

    up[1:Nyl, 1:Nxl] = u[1:Nyl, 1:Nxl] # Lower left
    up[1:Nyl, end-Nxu+1:end] = u[1:Nyl, end-Nxu+1:end] # Lower right
    up[end-Nyu+1:end, 1:Nxl] = u[end-Nyu+1:end, 1:Nxl] # Upper left
    up[end-Nyu+1:end, end-Nxu+1:end] = u[end-Nyu+1:end, end-Nxu+1:end] # Upper right
    return
end

function pad!(up, u, plan::rFFTPlans)
    Ny, Nx = size(u)

    Nxl = ceil(Int, Nx / 2)
    Nxu = floor(Int, Nx / 2)

    up[1:Ny, 1:Nxl] = u[1:Ny, 1:Nxl] # Lower left
    up[1:Ny, end-Nxu+1:end] = u[1:Ny, end-Nxu+1:end] # Lower right
    return
end

function unpad!(u, up, plan::FFTPlans)
    Ny, Nx = size(u)

    Nyl = ceil(Int, Ny / 2)
    Nyu = floor(Int, Ny / 2)
    Nxl = ceil(Int, Nx / 2)
    Nxu = floor(Int, Nx / 2)

    u[1:Nyl, 1:Nxl] = up[1:Nyl, 1:Nxl] # Lower left
    u[1:Nyl, end-Nxu+1:end] = up[1:Nyl, end-Nxu+1:end] # Lower right
    u[end-Nyu+1:end, 1:Nxl] = up[end-Nyu+1:end, 1:Nxl] # Upper left
    u[end-Nyu+1:end, end-Nxu+1:end] = up[end-Nyu+1:end, end-Nxu+1:end] # Upper right
    return
end

function unpad!(u, up, plan::rFFTPlans)
    Ny, Nx = size(u)

    Nxl = ceil(Int, Nx / 2)
    Nxu = floor(Int, Nx / 2)

    u[1:Ny, 1:Nxl] = up[1:Ny, 1:Nxl] # Lower left
    u[1:Ny, end-Nxu+1:end] = up[1:Ny, end-Nxu+1:end] # Lower right
    return
end

#-------------------------------- Differentiation ------------------------------------------

function diffX(field, SC::SpectralOperatorCache)
    SC.DiffX .* field
end

function diffY(field, SC::SpectralOperatorCache)
    SC.DiffY .* field
end

function laplacian(field, SC::SpectralOperatorCache)
    SC.Laplacian .* field
end

function diffXX(field, SC::SpectralOperatorCache)
    SC.DiffXX .* field
end

function diffYY(field, SC::SpectralOperatorCache)
    SC.DiffYY .* field
end

function poissonBracket(A, B, SC::SpectralOperatorCache)
    quadraticTerm(diffX(A, SC), diffY(B, SC), SC) - quadraticTerm(diffY(A, SC), diffX(B, SC), SC)
end

function solvePhi(field, SC::SpectralOperatorCache)
    SC.invLaplacian .* field
end

end