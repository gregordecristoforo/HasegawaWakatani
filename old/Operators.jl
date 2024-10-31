module Operators
export Laplacian

using Einsum
"""
Non-allocating function

dn - change in field\\
n - the current field\\
K - pre allocated K = -(k_x^2 + k_y^2)
"""
function Laplacian!(dn, n, K, D)
    @einsum dn[i,j] = D*n[i,j]*K[i,j]
end

"""
Laplacian Operator

n - current field
a - frequencies (parameter)
t - time (not used)
"""
function Laplacian(n, a, t)
    return Matrix{ComplexF64}([-100 * n[j, k] * (a[j]^2 + a[k]^2) for j in eachindex(a), k in eachindex(a)])
end


function PoissonBrackets(n, p, t)
    
end

"""
Short hand notation for laplacian Operator

n - current field
a - frequencies (parameter)
t - time (not used)
"""
function ∇(n, p, t)
    Laplacian(n, p, t)
end

end