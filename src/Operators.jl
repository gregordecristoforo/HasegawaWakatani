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

function Laplacian(a, n, p, t)
    [-D * n[j, k] * (a[j]^2 + a[k]^2) for j in eachindex(a), k in eachindex(a)]
end
end