using FFTW
using Plots
using PaddedViews

#Create range and function
x = -1:0.01:1-0.01
u = exp.(-10 * x .^ 2)

#Get modes and frequencies
u_hat = fft(u)
k = 2 * π * fftfreq(length(x), 1 / 0.01)

#Calculate derivatives and inverse fourier transform
du_hat = -im * k .* u_hat
du = ifft(du_hat)

#Multiply in real space and fourier transform back
v_hat = fft(u .* du)

function quadraticTerm(u, v)
    if size(u) != size(v)
        error("u and v must have the same size")
    end
    t = Tuple([-N÷4+1:N+N÷4 for N in size(u)])
    U = ifftshift(PaddedView(0, fftshift(u_hat), t)[t...])
    V = ifftshift(PaddedView(0, fftshift(du_hat), t)[t...])
    i = Tuple([1+N÷4:N+N÷4 for N in size(u)])
    ifftshift(fftshift(fft(ifft(U) .* ifft(V)))[i...])
end

v_hat = quadraticTerm(u_hat, du_hat)
plot(x, 1.5 * real(ifft(v_hat)))

plot(x, u)
plot(x, real(du))
plot(x, real(u .* du))
fft(u * du / dt) = fft(ifft(u_hat) * ifft(-im * k * u_hat))





t = Tuple([-N÷4+1:N+N÷4 for N in size(u)])
U = ifftshift(PaddedView(0, fftshift(u_hat), t))
i = Tuple([1+N÷4:N+N÷4 for N in size(u)])
v = real(ifft(U))
plot(v)

b = PaddedView(0, fftshift(u_hat), t)
plot(real(ifft(fftshift(b))))
plot(real(ifft(u_hat)))
using DSP
v = real(DSP.conv(u_hat, du_hat))[1:200]
plot(x, v)

using BenchmarkTools

@benchmark
t = Tuple([-3/2*N:3/2*N for N in size(a)])

function quadraticTerm(u, v)
    if size(u) != size(v)
        error("u and v must have the same size")
    end
    t = Tuple([-N÷4+1:N+N÷4 for N in size(u)])
    U = ifftshift(PaddedView(0, fftshift(u), t)[t...])
    V = ifftshift(PaddedView(0, fftshift(v), t)[t...])
    i = Tuple([1+N÷4:N+N÷4 for N in size(u)])
    display(plot(ifft(U) .* ifft(V)))
    fft(ifft(U) .* ifft(V))[i...]
end



ns = Tuple([3 * N ÷ 2 for N in size(u_hat)])
U = zeros(ns)

t = Tuple([-N÷4+1:N+N÷4 for N in size(u)])
U = PaddedView(0, u, t)[t...]

t = Tuple([-N÷4+1:N+N÷4 for N in size(u)])
U = ifftshift(PaddedView(0, fftshift(u_hat), t)[t...])
V = ifftshift(PaddedView(0, fftshift(du_hat), t)[t...])
i = Tuple([1+N÷4:N+N÷4 for N in size(u)])
v_hat = ifftshift(fftshift(fft(ifft(U) .* ifft(V)))[i...])

plot(real(ifft(v_hat)))

using PaddedViews

function quadraticTerm(u, v)
    if size(u) != size(v)
        error("u and v must have the same size")
    end
    t = Tuple([-N÷4+1:N+N÷4 for N in size(u)])
    U = ifftshift(PaddedView(0, fftshift(u), t)[t...])
    V = ifftshift(PaddedView(0, fftshift(v), t)[t...])
    i = Tuple([1+N÷4:N+N÷4 for N in size(u)])
    ifftshift(fftshift(fft(ifft(U) .* ifft(V)))[i...])
end

x = -1:0.01:1-0.01
u = Array(exp.(-10*x.^2))

plot(x,u)

u_hat = rfft(u)
k = 2 * π * rfftfreq(length(x),1/0.01)

du_hat = -im*k.*u_hat

v_hat = 1.5*quadraticTerm(u_hat, du_hat)

ifftPlot(v_hat)

ifftPlot(domain.x, domain.y, w0, st=:surface)

kx = prob.p["kx"]

dw = -im*kx'.*w0
ifftPlot(dw, st=:surface)

v2 = quadraticTerm(w0, dw)
ifftPlot(v2, st=:surface)

p = prob.p
du = im*Matrix([(p["kx"][j])*w0[i,j] for i in eachindex(p["kx"]), j in eachindex(p["ky"])])
ifftPlot(du, st=:surface)

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