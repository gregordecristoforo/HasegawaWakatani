using FFTW
using Plots

x = -1:0.01:1-0.01
u = Array(exp.(-10*x.^2))

plot(x,u)

u_hat = rfft(u)
k = 2 * π * rfftfreq(length(x),1/0.01)

du_hat = -im*k.*u_hat

using PaddedViews
dud = Array(PaddedView(0, du_hat, (151, )))

du = irfft(dud, 300)
plot(real(du))

plot(real(du_hat))
plot(x,real(du))
