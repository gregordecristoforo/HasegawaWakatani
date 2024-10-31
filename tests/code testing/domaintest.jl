using Plots
domain = Domain(16)
scatter(getDomainFrequencies(domain) ./ (2 * pi))

k_x, k_y = getDomainFrequencies(domain)
x = domain.x
for k in k_x
    display(plot(x, sin.(k * x)))
end