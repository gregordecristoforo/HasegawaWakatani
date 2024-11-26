using Plots
domain = Domain(16)
scatter(getDomainFrequencies(domain) ./ (2 * pi))

k_x, k_y = getDomainFrequencies(domain)
x = domain.x
for k in k_x
    display(plot(x, sin.(k * x)))
end


domain.x[end] + domain.dx
domain.x[1]
size(domain.x)

1/(128-1)
domain.dx

domain.x[2] - domain.x[1]
