## Run all (alt+enter)
include("../src/HasagawaWakatini.jl")

## Run scheme test for Burgers equation
#domain = Domain(512, 512, 200, 100, anti_aliased=false)
#domain = Domain(256, 256, 50, 50, anti_aliased=false, realTransform=false)
domain = Domain(1024, 1024, 50, 50, anti_aliased=false)
u0 = gaussian.(domain.x', domain.y, A=1, B=1, l=1)

# Linear operator
function L(u, d, p, t)
    D_θ = p["kappa"] * diffusion(u, d)
    D_Ω = p["nu"] * diffusion(u, d)
    [D_θ;;; D_Ω]
end

# Non-linear operator
function N(u, d, p, t)
    θ = u[:, :, 1]
    Ω = u[:, :, 2]
    ϕ = solvePhi(Ω, d)
    dθ = -poissonBracket(ϕ, θ, d)
    dΩ = -poissonBracket(ϕ, Ω, d)
    dΩ -= diffY(θ, d)
    return [dθ;;; dΩ]
end

# Parameters
parameters = Dict(
    "nu" => 1e-2,
    "kappa" => 1e-2
)

# Time interval
t_span = [0, 20]

# The problem
prob = SpectralODEProblem(L, N, domain, [u0;;; zero(u0)], t_span, p=parameters, dt=1e-3)

# The output
output = Output(prob, 1000, [progressDiagnostic])

## Solve and plot
sol = spectral_solve(prob, MSS3(), output)

## Recreate Garcia et al. plots
display(heatmap(sol.u[5][:,:,1], levels=10, aspect_ratio=:equal))
display(heatmap(sol.u[10][:,:,1], levels=10, aspect_ratio=:equal))
display(heatmap(sol.u[15][:,:,1], levels=10, aspect_ratio=:equal))
display(heatmap(sol.u[end][:,:,1], levels=10, aspect_ratio=:equal))
display(heatmap(sol.u[5][:,:,2], levels=10, aspect_ratio=:equal, color=:jet))
display(heatmap(sol.u[10][:,:,2], levels=10, aspect_ratio=:equal, color=:jet))
display(heatmap(sol.u[15][:,:,2], levels=10, aspect_ratio=:equal, color=:jet))
display(heatmap(sol.u[end][:,:,2], levels=10, aspect_ratio=:equal, color=:jet))

## Save data
using JLD
save("reproduced data Garcia 2005 PoP.jld", "data", sol.u)

##

n = sol.u[end-1][:,:,1]
H = [n[y,x]*domain.x[x] for y in eachindex(domain.y), x in eachindex(domain.x)]



contour(sol.u[1][:,:,1], aspect_ratio=:equal)
contour(sol.u[end-1][:,:,1], aspect_ratio=:equal)

data = sol.u[1][:,:,1]
data_hat = fft(data)
DiffX = im * domain.kx'
df_hat =  DiffX.*data_hat
df = real(ifft(df_hat))
maximum(df)*1e-2/domain.dx
contourf(domain,df)

DiffY = im * domain.ky
df2_hat = DiffY .* data_hat
df2 = real(ifft(df2_hat))
maximum(df2)*1e-2/domain.dx
contourf(df2)

plotlyjsSurface(z=df2)


X_COM = zeros(length(A))
for i in range(1,19)
    n = A[i][:,:,1]
    X_COM[i] = sum(n[2:end,2:end].*domain.x'[2:end])/sum(n[2:end])
end

plot(X_COM[1:19])

contour(A[end][:,:,1].-sol.u[end][:,:,1], aspect_ratio=:equal)




# Check the transforms
ic = [u0;;; zero(u0)]
ic_hat = transform(ic, domain.transform.FT)
ic_2 = transform(ic_hat, domain.transform.iFT)

ic_hat[:,prob.domain.Nx÷2+1,:] .= 0

df_hat = prob.f(ic_hat, prob.domain, prob.p, 0)
df = transform(df_hat, domain.transform.iFT)

plotlyjsSurface(z=sol.u[10][:,:,2])












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


D_θ = domain.SC.DiffY .+ domain.SC.Laplacian
D_Ω = domain.SC.Laplacian
D = [D_θ;;; D_Ω]

a = [1, 1]
a .* D_Ω

D .* ic_hat

B = [[1;; 2];;; [1;; 2]]

B .* B

B .+ B


D
dt = 0.01
c = @. (1 - D * dt)^-1

L(1, domain, parameters, 0)
zero(D)

ic = [u0;;; zero(u0)]
ic_hat = rfft(ic)
vec = f(ic_hat, domain, parameters, 0)

transform(ic, domain.transform.FT)

ic
