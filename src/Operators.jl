module Operators
export Laplacian

"""
Non-allocating function

dn - change in field\\
n - the current field @ time t\\
p - parameters\\
t - time
"""
function Laplacian!(dn, n, p, t)
    v = copy(n)
    dn = [-D * v[j, k] * (a[j]^2 + a[k]^2) for j in eachindex(a), k in eachindex(a)]
    nothing
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