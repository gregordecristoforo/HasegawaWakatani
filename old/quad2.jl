using FFTW
using Plots
using PaddedViews
using BenchmarkTools

@benchmark
t = Tuple([-3/2*N:3/2*N for N in size(a)])


using Plots
plot(D.x, D.y, u, st=:surface)

uhat = rfft(u)

ohat = DiffXX(uhat, D.SC)
mhat = PoissonBracket(uhat, ohat, D.SC)

plot(D.x, D.y, real(ifft(ohat)), st=:surface, camera=(90, 0))



# -------------------------------------- New code ------------------------------------------

## Test conditions
include("../src/utilities.jl")
domain = Domain(128, 1)
u0 = initial_condition(gaussian, domain, l=0.08)
u0_hat = fft(u0)
r0_hat = domain.transformPlans.FT * u0

## Testing

plotFrequencies(quadraticTerm(A, A, domain.SC, true))


surface(domain.transformPlans.iFT * quadraticTerm(r0_hat, diffX(r0_hat, domain.SC), domain.SC, true))
surface(domain.transformPlans.iFT * quadraticTerm(r0_hat, diffX(r0_hat, domain.SC), domain.SC, false))

surface(domain.transformPlans.iFT * A)

maximum(domain.transformPlans.iFT * quadraticTerm(r0_hat, r0_hat, domain.SC, true))


A = zero(r0_hat)
pad!(domain.SC.up, r0_hat)
unpad!(A, domain.SC.up)
include("../src/diagnostics.jl")
plotFrequencies(r0_hat)
plotFrequencies(A)
plotFrequencies(fftshift(domain.SC.QTp, 2))

surface(real(domain.SC.QTPlans.iFT * domain.SC.up))






# ------------------------------- Old code -------------------------------------------------
# Quadratic terms interface 
function quadraticTerm(u, v, domain; padded=true)
    if size(u) != size(v)
        error("u and v must have the same size")
    end

    if domain.real
        rQT(u, v, d, padded=padded)
    else
        QT(u, v, d, padded=padded)
    end
end

function QT(u, v, d::Domain, padded=padded)
    if padded
        t = Tuple([-N÷4+1:N+N÷4 for N in size(u)])
        U = ifftshift(PaddedView(0, fftshift(u), t)[t...])
        V = ifftshift(PaddedView(0, fftshift(v), t)[t...])
        i = Tuple([1+N÷4:N+N÷4 for N in size(u)])
        1.5 * ifftshift(fftshift(fft(ifft(U) .* ifft(V)))[i...])
    else
        fft(ifft(u) .* ifft(v))
    end
end

##





function pad!(up, u, realTransform=true)
    Ny, Nx = size(u)

    Nxl = ceil(Int, Nx / 2)
    Nxu = floor(Int, Nx / 2)
    Nyl = ceil(Int, Ny / 2)
    Nyu = floor(Int, Ny / 2)

    if realTransform
        up[1:Ny, 1:Nxl] = u[1:Ny, 1:Nxl] # Lower left
        up[1:Ny, end-Nxu+1:end] = u[1:Ny, end-Nxu+1:end] # Lower right
        return
    else
        up[1:Nyl, 1:Nxl] = u[1:Nyl, 1:Nxl] # Lower left
        up[1:Nyl, end-Nxu+1:end] = u[1:Nyl, end-Nxu+1:end] # Lower right
        up[end-Nyu+1:end, 1:Nxl] = u[end-Nyu+1:end, 1:Nxl] # Upper left
        up[end-Nyu+1:end, end-Nxu+1:end] = u[end-Nyu+1:end, end-Nxu+1:end] # Upper right
        return
    end
end

function unpad!(u, up, realTransform=true)
    Ny, Nx = size(u)

    Nyl = ceil(Int, Ny / 2)
    Nyu = floor(Int, Ny / 2)
    Nxl = ceil(Int, Nx / 2)
    Nxu = floor(Int, Nx / 2)

    if realTransform
        u[1:Ny, 1:Nxl] = up[1:Ny, 1:Nxl] # Lower left
        u[1:Ny, end-Nxu+1:end] = up[1:Ny, end-Nxu+1:end] # Lower right
        return
    else
        u[1:Nyl, 1:Nxl] = up[1:Nyl, 1:Nxl] # Lower left
        u[1:Nyl, end-Nxu+1:end] = up[1:Nyl, end-Nxu+1:end] # Lower right
        u[end-Nyu+1:end, 1:Nxl] = up[end-Nyu+1:end, 1:Nxl] # Upper left
        u[end-Nyu+1:end, end-Nxu+1:end] = up[end-Nyu+1:end, end-Nxu+1:end] # Upper right
        return
    end
end


## Test code
domain = Domain(6, 10, 1, 1, realTransform=true)

H = rand(domain.Ny, domain.Nx)
m = domain.transformPlans.FT * H

domain.SC.QTPlans.iFT * domain.SC.up