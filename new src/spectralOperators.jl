struct SpectralOperatorCoefficents
    DiffX::Array{ComplexF64}
    DiffY::Array{ComplexF64}
    DiffXX::Array{ComplexF64}
    DiffYY::Array{ComplexF64}
    Laplacian::Array{ComplexF64}

    function SpectralOperatorCoefficents(kx, ky)
        DiffX = im * kx
        DiffY = im * ky'
        DiffXX = -kx .^ 2
        DiffYY = -ky' .^ 2
        Laplacian = DiffXX .+ DiffYY
        new(DiffX, DiffY, DiffXX, DiffYY, Laplacian)
    end
end

function DiffX(field, SC::SpectralOperatorCoefficents)
    SC.DiffX .* field
end

function DiffY(field, SC::SpectralOperatorCoefficents)
    SC.DiffY .* field
end

function Diffusion(field, SC::SpectralOperatorCoefficents)
    SC.Laplacian .* field
end

function DiffXX(field, SC::SpectralOperatorCoefficents)
    SC.DiffXX .* field
end

function DiffYY(field, SC::SpectralOperatorCoefficents)
    SC.DiffYY .* field
end

using PaddedViews

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

function PoissonBracket(A, B, SC::SpectralOperatorCoefficents, padded=true)
    quadraticTerm(DiffX(A, SC), DiffY(B, SC)) - quadraticTerm(DiffY(A, SC), DiffX(B, SC))
end