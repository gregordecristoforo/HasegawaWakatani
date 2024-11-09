
module SpectralOperators
export SpectralOperatorCoefficents, diffX, diffY, diffXX, diffYY, diffusion, solvePhi, poissonBracket, quadraticTerm

struct SpectralOperatorCoefficents
    DiffX::Array{ComplexF64}
    DiffY::Array{ComplexF64}
    DiffXX::Array{ComplexF64}
    DiffYY::Array{ComplexF64}
    Laplacian::Array{ComplexF64}

    function SpectralOperatorCoefficents(kx, ky)
        DiffX = im * kx'
        DiffY = im * ky
        DiffXX = -kx' .^ 2
        DiffYY = -ky .^ 2
        Laplacian = -kx' .^ 2 .- ky .^ 2
        new(DiffX, DiffY, DiffXX, DiffYY, Laplacian)
    end
end

using PaddedViews
using FFTW

# Calculate quadratic terms 
function quadraticTerm(u, v, padded=true)
    if size(u) != size(v)
        error("u and v must have the same size")
    end
    if padded
        t = Tuple([-N÷4+1:N+N÷4 for N in size(u)])
        U = ifftshift(PaddedView(0, fftshift(u), t)[t...])
        V = ifftshift(PaddedView(0, fftshift(v), t)[t...])
        i = Tuple([1+N÷4:N+N÷4 for N in size(u)])
        1.5 * ifftshift(fftshift(fft(ifft(U) .* ifft(V)))[i...])
    else
        fft(ifft(u) .* ifft(v))
    end
end

function diffX(field, SC::SpectralOperatorCoefficents)
    SC.DiffX .* field
end

function diffY(field, SC::SpectralOperatorCoefficents)
    SC.DiffY .* field
end

function diffusion(field, SC::SpectralOperatorCoefficents)
    SC.Laplacian .* field
end

function diffXX(field, SC::SpectralOperatorCoefficents)
    SC.DiffXX .* field
end

function diffYY(field, SC::SpectralOperatorCoefficents)
    SC.DiffYY .* field
end

function poissonBracket(A, B, SC::SpectralOperatorCoefficents, padded=true)
    quadraticTerm(diffX(A, SC), diffY(B, SC)) - quadraticTerm(diffY(A, SC), diffX(B, SC))
end

function solvePhi(field, SC::SpectralOperatorCoefficents)
    phi_hat = field ./ SC.Laplacian
    phi_hat[1] = 0 # First entry will always be NaN
    return phi_hat
end
end