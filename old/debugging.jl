# Testing out 2D convolution
domain = Domain(64, 1)
u0 = initial_condition(gaussianWallY, domain, l=0.08)
du = domain.transform.iFT * diffY(domain.transform.FT * u0, domain)

using DSP
surface(domain.transform.iFT * quadraticTerm(domain.transform.FT * u0, diffY(domain.transform.FT * u0, domain), domain))
plotlyjsSurface(z=conv(domain.transform.FT * u0, domain.transform.FT * du))
surface(conv(du, u0))
surface(irfft(conv(domain.transform.FT * u0, domain.transform.FT * du), 128))