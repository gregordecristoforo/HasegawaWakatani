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

#function 