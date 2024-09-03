#TODO implement way to get velocity of field iku?
function v(u)
end

#Calculate max cfl in x direction Pseudocode #TODO implement properly
function cflx(u)
    max(abs(v(u))) * dt / dx
end

function compare(x, y, A::Matrix, B::Matrix)
    println(norm(A - B))
    #plot(x, A)
end

# Uses the Heat equation to test at the moment
function testTimestepConvergence(scheme, prob, analyticalSolution, timesteps)

    #Calculate analyticalSolution
    u = analyticalSolution(prob)

    #Initialize storage
    residuals = zeros(size(timesteps))

    for (i, dt) in enumerate(timesteps)
        #Change timestep of spectralODEProblem
        prob.dt = dt
        #Calculate approximate solution
        _, uN = scheme(prob, output=Nothing, singleStep=false)
        residuals[i] = norm(ifft(uN) - ifft(u))
    end

    #Plot residuals vs. time
    plot(timesteps, residuals, xaxis=:log, yaxis=:log, xlabel="dt", ylabel="||u-u_a||")
end

#
function testResolutionConvergence(scheme, prob, initialField, analyticalSolution, resolutions)
    cprob = deepcopy(prob)
    residuals = zeros(size(resolutions))

    for (i, N) in enumerate(resolutions)
        domain = Domain(N, 4)
        updateDomain!(cprob, domain)
        updateInitalField!(cprob, initialField)
        #prob = SpectralODEProblem(prob.f, domain, prob.u0, prob.tspan, p=prob.p, dt=prob.dt)
        #prob = SpectralODEProblem(prob.f, prob.domain, fft(initialField(prob.domain, prob.p)), prob.tspan, p = prob.p, dt=prob.dt)
        println(size(cprob.u0))
        u = analyticalSolution(cprob)

        _, uN = scheme(cprob, output=Nothing, singleStep=false)
        # Scaled residual to compensate for increased resolution
        residuals[i] = norm(ifft(u) - ifft(uN)) / (domain.Nx * domain.Ny)
        println(maximum(real(ifft(u))) - maximum(real(ifft(uN))))
    end

    display(plot(resolutions, residuals, xaxis=:log2, yaxis=:log))#, st=:scatter))
    display(plot(resolutions, resolutions .^ -2, xaxis=:log2, yaxis=:log))#, st=:scatter))
end

testResolutionConvergence(mSS1Solve, prob, gaussianBlob, HeatEquationAnalyticalSolution, [16, 32, 64, 128, 256])