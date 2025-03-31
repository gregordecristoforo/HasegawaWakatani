## Run all (alt+enter)
include("../src/HasegawaWakatini.jl")

## Run scheme test for Burgers equation
#domain = Domain(512, 512, 200, 100, anti_aliased=false)
domain = Domain(128, 128, 100, 100, anti_aliased=false)
u0 = 1 .+ gaussian.(domain.x', domain.y, A=1, B=0, l=3)
u0 = 1 .+ gaussian.(domain.x', domain.y, A=20, B=0, l=3)
u0 = log.(u0)

# Linear operator
function L(u, d, p, t)
    D_n = p["D_n"] .* hyper_diffusion(u, d) .- p["C"] * u
    D_Ω = p["D_Ω"] .* hyper_diffusion(u, d)
    [D_n;;; D_Ω]
end

# Non-linear operator, linearized
function N(u, d, p, t)
    n = u[:, :, 1]
    Ω = u[:, :, 2]
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
    [dn;;; dW]
end

# # Non-linear operator, fully non-linear
# function N(u, d, p, t)
#     n = u[:, :, 1]
#     W = u[:, :, 2]
#     phi = solvePhi(W, d)
#     dn = -poissonBracket(phi, n, d)
#     dn += p["g"] * quadraticTerm(n, diffY(n, d), d)
#     dn += -p["g"] * diffY(n, d)
#     dn += -p["sigma"] * n
#     dW = -poissonBracket(phi, W, d)
#     #dW += -p["g"] * diffY(spectral_log(n, d), d)
#     dW += -p["g"] * quadraticTerm(reciprocal(n, d), diffY(n, d), d)
#     dW += n
#     dW += -quadraticTerm(n, spectral_exp(phi, d), d)
#     [dn;;; dW]
# end

# Parameters
parameters = Dict(
    "nu" => 1e-2,
    "g" => 1e-3, #1e-2
    "sigma" => 5e-4,
)

t_span = [0, 0.1]

prob = SpectralODEProblem(f, domain, [u0;;; zero(u0)], t_span, p=parameters, dt=1e-5)

output = Output(prob, 1000, [plot_nDiagnostic, progressDiagnostic])

## Solve and plot
sol = spectral_solve(prob, MSS3(), output)