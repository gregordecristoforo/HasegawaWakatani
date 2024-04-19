# Numerical integration method

module Timestepper
export timeStep

# Butcher tableau
B = [0 0 0 0;
    1/2 1/2 0 0;
    1 -1 2 0;
    0 1/6 2/3 1/6]

k = [zeros(Complex{Float64}, 64, 64) for i in 1:3]
q = 3

function timeStep(y, p, t, dt, rhs)
    k[1] = rhs(y, p, t)
    for i in 2:q
        k[i] = rhs(y + dt * sum((B[i, j+1] * k[j] for j in 1:i-1)), p, t + B[i, 1] * dt)
    end

    return y + dt * sum((B[end, i+1] * k[i] for i in 1:q))
end

end
#f/m = dv/dt, v = dx/dt