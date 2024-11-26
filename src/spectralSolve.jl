#include("spectralODEProblem.jl")
include("schemes.jl")
include("fftutilities.jl")

export spectralSolve

function spectral_solve(prob::SpectralODEProblem, scheme=MSS3(), output=Nothing())
    # Initialize cache
    cache = get_cache(prob, scheme)

    t = first(prob.tspan)
    tend = last(prob.tspan)
    dt = prob.dt
    domain = prob.domain
    step = 0

    # Initialize output 
    U = zeros(size(prob.u0)..., floor(Int, tend / dt / 100))

    while t <= tend
        perform_step!(cache, prob, t)
        # TODO remove
        #cache.u[domain.Ny÷2+1, :, :] .= 0
        #cache.u[:, domain.Nx÷2+1, :] .= 0
        step += 1
        t += dt
        # TODO implement output
        if step % 100 == 0
            U[:, :, :, step÷100] = real(multi_ifft(cache.u, domain.transform))

            #n = u[:, :, 1]
            #W = u[:, :, 2]
            #println(size(n), size(W))
            #display(surface(domain.x, domain.y, n, xlabel="x", ylabel="y"))
            #display(contourf(W))
            #cfl = maximum(real(multi_ifft(diffY(cache.u, domain), domain.transform))) * dt / domain.dy
            #display(plot(domain.y, real(multi_ifft(cache.u, domain.transform)), title="t=$t, cfl=$cfl"))
            #display(contourf(domain, real(multi_ifft(cache.u, domain.transform)), title="t=$t, cfl=$cfl"))
        end
    end

    #return t - dt, multi_ifft(cache.u, domain.transform)
    return U
end