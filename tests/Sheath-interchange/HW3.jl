## Run all (alt+enter)
include("../src/HasegawaWakatini.jl")

## Run scheme test for Burgers equation
#domain = Domain(512, 512, 200, 100, anti_aliased=false)
domain = Domain(256, 256, 100, 100, anti_aliased=false)
u0 = 1 .+ gaussian.(domain.x', domain.y, A=20, B=0, l=3)
u0 = log.(u0)

function f(u, d, p, t)
    n = u[:, :, 1]
    W = u[:, :, 2]
    phi = solvePhi(W, d)
    dn = -poissonBracket(phi, n, d)
    dn += p["g"] * diffY(phi, d)
    dn += -p["g"] * diffY(n, d)
    dn += -p["sigma"] * n
    dW = -poissonBracket(phi, W, d)
    #dW += -p["g"] * diffY(spectral_log(n, d), d)
    dW += -p["g"] * diffY(n, d)
    dW += n
    dW += -quadraticTerm(n, spectral_exp(phi, d), d)
    
    println(maximum(irfft(-p["g"] * diffY(n, d), d.Ny)))
    
    [dn;;; dW]
end

# Parameters
parameters = Dict(
    "nu" => 1e-2,
    "g" => 1e-2,
    "sigma" => 5e-4,
)

t_span = [0, 0.1]

prob = SpectralODEProblem(f, domain, [u0;;; zero(u0)], t_span, p=parameters, dt=1e-5)

output = Output(prob, 1000, [plot_nDiagnostic, progressDiagnostic])

## Solve and plot
sol = spectral_solve(prob, MSS3(), output)

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