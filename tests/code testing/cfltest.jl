## Include all modules
include("../../src/HasagawaWakatini.jl")

## Test implementation of vExB
domain = Domain(128, 1)
phi = initial_condition(sinusoidal, domain)
plot(domain, phi)
n0 = initial_condition(gaussian, domain, l=0.08)
plot(domain, n0)

# Transform
n0_hat = rfft(n0)
phi_hat = rfft(phi)

u = [phi_hat;;; phi_hat]
v_x, v_y = vExB(u, domain)

# Plot velocity fields
surface(domain, v_x)
surface(domain, v_y)

# Check maxCFL in x and y direction
maxCFLx(u, domain, 0.01)
maxCFLy(u, domain, 0.01)

# Testing speed field
v = sqrt.(v_x .^ 2 + v_y .^ 2)
argmax(v)
plot(domain, v)