using PaddedViews

function quadraticTerm(u, v, padded=true)
    if size(u) != size(v)
        error("u and v must have the same size")
    end
    if padded
        t = Tuple([-N÷4+1:N+N÷4 for N in size(u)])
        U = ifftshift(PaddedView(0, fftshift(u), t)[t...])
        V = ifftshift(PaddedView(0, fftshift(v), t)[t...])
        i = Tuple([1+N÷4:N+N÷4 for N in size(u)])
        1.5*ifftshift(fftshift(fft(ifft(U) .* ifft(V)))[i...])
    else
        fft(ifft(u) .* ifft(v))
    end
end

function quadraticTerm2(u, v)
    if size(u) != size(v)
        error("u and v must have the same size")
    end
    
end

D = Domain(64, 64, 1, 1, real=false)
u = @. exp(-10*(D.x^2 + D.y'^2))

using Plots
plot(D.x, D.y, u, st=:surface)

uhat = rfft(u)
vhat = DiffX(uhat, D.SC) + DiffY(uhat, D.SC)
v = irfft(vhat, D.Nx)

what = quadraticTerm(uhat, vhat)
w = irfft(what, D.Nx)
w = real(ifft(what))

plot(D.x, D.y, w, st=:surface)#, camera=(90,0))

ohat = DiffXX(uhat, D.SC)
mhat = PoissonBracket(uhat, ohat, D.SC)

plot(D.x, D.y, real(ifft(ohat)), st=:surface, camera=(90,0))