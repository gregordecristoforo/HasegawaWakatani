## Run all (alt+enter)
include("../src/HasagawaWakatini.jl")

## Run scheme test for Burgers equation
#domain = Domain(512, 512, 200, 100, anti_aliased=false)
domain = Domain(256, 256, 50, 50, anti_aliased=false)
u0 = 1 .+ gaussian.(domain.x', domain.y, A=1, B=0, l=1)

contourf(u0)

function f(u, d, p, t)
    θ = u[:, :, 1]
    Ω = u[:, :, 2]
    ϕ = solvePhi(Ω, d)
    dθ = -poissonBracket(ϕ, θ, d)
    dΩ = -poissonBracket(ϕ, Ω, d)
    dΩ = -diffY(θ, d)
    return [dθ;;; dΩ]
end

# Parameters
parameters = Dict(
    "nu" => 1e-2
)

t_span = [0, 5]

prob = SpectralODEProblem(f, domain, [u0;;; zero(u0)], t_span, p=parameters, dt=1e-4)

output = Output(prob, 1000, [plot_nDiagnostic, progressDiagnostic])

## Solve and plot
sol = spectral_solve(prob, MSS3(), output)

A = [u0, u0]
b = [0.2, 0.2]

fft([u0, u0])

contourf(domain, sol.u[end][:, :, 2])
contourf(domain, output.u[7][:, :, 1])

surface(domain, uend)
contourf(domain, uend[:, :, 1])
xlabel!("x")

plotlyjsSurface(z=uend)
plotlyjsSurface(z=uend[:, :, 1])







## Debug

u0_hat = domain.transform.FT * u0
f_hat = f([u0_hat;;; u0_hat], domain, parameters, 0)
F = transform(f_hat, domain.transform.iFT)
plotlyjsSurface(z=F[:, :, 1])
plotlyjsSurface(z=F[:, :, 2])

plotlyjsSurface(z=(1) ./ u0)


s = domain.transform.iFT * inverse(prob.u0_hat[:, :, 1], domain)
plotlyjsSurface(z=s)


#nu = [1e-2, 1e-2]
#A = [u0;;; zero(u0)]
#nu.*A

# Calculate COM 
Θ = sol.u[end][:, :, 1]

sum(x .* Θ) / sum(Θ)

sum(domain.y .* Θ)
sum(Θ)

x = domain.x .^ 2 .+ domain.y' .^ 2

# Calculate 1d field

for i in eachindex(sol.u)
    display(plot!(sum(sol.u[i][:, :, 1], dims=1)' ./ domain.Ly))
end

plot(sum(Θ, dims=2))
