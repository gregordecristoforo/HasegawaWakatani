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