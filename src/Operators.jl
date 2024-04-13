module Operators
export Laplacian

# Non-allocating function
# dn - change in field
# n - the current field @ time t  
# p - parameters
# t - time
function Laplacian!(dn, n, p, t)
    v = copy(n)
    dn = [-D * v[j, k] * (a[j]^2 + a[k]^2) for j in eachindex(a), k in eachindex(a)]
    nothing
end

function Laplacian(t::Float64, n::Matrix{ComplexF64})
    a = FFTW.fftfreq(size(n)[1])
    return [-D * n[j, k] * (a[j]^2 + a[k]^2) for j in eachindex(a), k in eachindex(a)]
end

function ∇(a, n, p, t)
    Laplacian(a, n, p, t)
end

end