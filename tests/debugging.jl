#Array functions
eltype(kappa)
length(kappa)
ndims(kappa)
size(kappa)
axes(kappa)
axes(kappa, 1)
axes(kappa, 2)
eachindex(kappa)
strides(kappa)

#Use comprehension
collect(1:4)
v = 1:2
B = reshape(collect(1:16), (2, 2, 2, 2))

function Diffusion(field, domain, nu)
    @. domain.SC.Laplacian * field
end

D = Domain(64, 2, 1, 1)

f = ones(64, 2)
fhat = rfft(f)

Diffusion(fhat, D, 0.1)