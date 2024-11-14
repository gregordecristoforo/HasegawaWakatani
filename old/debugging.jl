#Array functions
eltype(kappa)
length(kappa)
ndims(kappa)
size(kappa)
axes(kappa)
axes(kappa, 1)
axes(kappa, 2)
eachindex(kappa)
strides(kappa)

using Plots

plot(domain.x, domain.y, real(ifft(w0)), st=:surface)
plot(domain.x, domain.y, real(ifft(u)), st=:surface)
plot(domain.x, real(ifft(u))[1, :])
xlabel!("x")
ylabel!("y")

ifftPlot(domain.x, domain.y, u, title="Time step $(dt)", st=:surface)
ifftPlot(domain.x, domain.y, HeatEquationAnalyticalSolution(n0, 2, -prob.p["k2"], 0.1), title="Time step $(dt)", st=:surface)
ifftPlot(domain.x, domain.y, HeatEquationAnalyticalSolution(n0, 2, -prob.p["k2"], 0.1) - u)
##

prob = SpectralODEProblem(f, domain, n0, [0, 0.1], p=parameters, dt=dt)
testTimestepConvergence(mSS3Solve, prob, HeatEquationAnalyticalSolution, [0.1, 0.01, 0.001, 0.0001, 0.00001])

prob.dt = 0.001
testResolutionConvergence(mSS3Solve, prob, gaussianBlob, HeatEquationAnalyticalSolution, [16, 32, 64, 128, 256, 512, 1024])

plot(domain.x, domain.y, real(ifft(HeatEquationAnalyticalSolution(prob))), st=:surface)

updateDomain!(prob, domain)

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