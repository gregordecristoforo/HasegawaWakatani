#Array functions
eltype(kappa)
strides(kappa)

prob = SpectralODEProblem(f, domain, n0, [0, 0.1], p=parameters, dt=dt)
testTimestepConvergence(mSS3Solve, prob, HeatEquationAnalyticalSolution, [0.1, 0.01, 0.001, 0.0001, 0.00001])

prob.dt = 0.001
testResolutionConvergence(mSS3Solve, prob, gaussianBlob, HeatEquationAnalyticalSolution, [16, 32, 64, 128, 256, 512, 1024])

using HDF5

A = Vector{Int}(1:10)
h5write("bar.h5", "fun", A .+ 1)

h5open("bar.h5", "w") do file
    g = create_group(file, "mygroup")
end

h5read("bar.h5", "fun")
h5writeattr("bar.h5", "fun", Dict("c" => "value for metadata parameter c", "d" => "metadata d"))
h5readattr("bar.h5", "fun")

# Testing out 2D convolution
domain = Domain(64, 1)
u0 = initial_condition(gaussianWallY, domain, l=0.08)
du = domain.transform.iFT * diffY(domain.transform.FT * u0, domain)

using DSP
surface(domain.transform.iFT * quadraticTerm(domain.transform.FT * u0, diffY(domain.transform.FT * u0, domain), domain))
plotlyjsSurface(z=conv(domain.transform.FT * u0, domain.transform.FT * du))
surface(conv(du, u0))
surface(irfft(conv(domain.transform.FT * u0, domain.transform.FT * du), 128))