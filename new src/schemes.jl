export mSS1Solve, mSS2Solve, mSS3Solve

function forwardEuler(f, y0, t)

    dt = t[2] - t[1]
    yp = y0
    y = similar(yp)
    for i in t
        y = yp + dt * f(yp)
        yp = y
    end

    return y
end

function mixedStifflyStable(f, y0, t, p, order=1)
    a = [1 3//2 11//6
        1 2 3
        0 -1//2 -3//2
        0 0 1//3]

    b = [1 2 3
        0 -1 -3
        0 0 1]

    kappa = p["kappa"]
    u = Array{ComplexF64,2}
    alpha = [1, 1, 0]
    beta = [1, 0, 0]
    u_step

    for (i, u_row) = enumerate(eachrow(u))
        u_step = @. alpha[i] * u_row + kappa * Laplacian(u_row) + beta * f(u_row)
    end

    return u_step
end

include("spectralODEProblem.jl")

function mSS1Solve(prob::SpectralODEProblem; output=Nothing, singleStep=false)
    #Coefficents are all 1

    #Cache
    u1 = prob.u0
    u2 = zero(prob.u0)

    #dt
    dt = prob.dt

    #Get k2
    k2 = prob.p["k2"]

    #Read parameter
    nu = prob.p["nu"]

    #Calculate "scaling" factor
    c = @. (1 + nu * k2 * dt)^-1

    t = prob.tspan[1]

    #Perform steps untill achieved t
    while t <= prob.tspan[2]
        #Step
        t += dt
        u2 = c .* (u1 + dt * f(u1, prob.p, t - dt))
        u1 = u2
        if singleStep
            break
        end
    end

    return t, u2
end

function mSS2Solve(prob::SpectralODEProblem; output=Nothing, singleStep=false)
    #Coefficents
    g0 = 3 // 2
    a2 = 2
    a1 = -1 // 2
    b2 = 2
    b1 = -1

    #Cache
    u1 = prob.u0
    t, u2 = mSS1Solve(prob, output=output, singleStep=true)
    u3 = zero(prob.u0)

    #dt
    dt = prob.dt

    #Get k2
    k2 = prob.p["k2"]

    #Read parameter
    nu = prob.p["nu"]

    #Calculate coeff
    c = @. (g0 + nu * k2 * dt)^-1

    #Perform steps untill achieved t
    while t <= prob.tspan[2]
        t += dt
        #Calculate explicit and implicit terms
        u_i = a1 * u1 + a2 * u2
        u_e = b1 * f(u1, prob.p, t - dt) + b2 * f(u2, prob.p, t - 2 * dt)
        #Step
        u3 = c .* (u_i + dt * u_e)
        #Swap memory 
        u1, u2 = u2, u3
        if singleStep
            break
        end
    end
    t, u3
end

function mSS3Solve(prob::SpectralODEProblem; output=Nothing, singleStep=false)
    #Coefficents
    g0 = 11 // 6
    a3 = 3
    a2 = -3 // 2
    a1 = 1 // 3
    b3 = 3
    b2 = -3
    b1 = 1

    #Cache
    u1 = prob.u0
    t, u2 = mSS1Solve(prob, output=output, singleStep=true)
    t, u3 = mSS2Solve(prob, output=output, singleStep=true)
    u4 = zero(prob.u0)

    #dt
    dt = prob.dt

    #Get k2
    k2 = prob.p["k2"]

    #Read parameter
    nu = prob.p["nu"]

    #Calculate coeff
    c = @. (g0 + nu * k2 * dt)^-1

    #Perform steps untill achieved t
    while t <= prob.tspan[2]
        #Calculate explicit and implicit terms
        t += dt
        u_i = a1 * u1 + a2 * u2 + a3 * u3
        u_e = b1 * f(u1, prob.p, t - dt) + b2 * f(u2, prob.p, t - 2 * dt) + b3 * f(u3, prob.p, t - 3 * dt)

        #Step
        u4 = c .* (u_i + dt * u_e)
        #Swap memory 
        u1, u2, u3 = u2, u3, u4
        if singleStep
            break
        end
    end


    t, u4
end
