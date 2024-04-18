# Numerical integration method

module Timestepper
export timeStep

# Butcher tableau
B = [0 0 0 0;
    1/2 1/2 0 0;
    1 -1 2 0;
    0 1/6 2/3 1/6]

k = [0.0, 0.0, 0.0]
q = 3

function timeStep(t, y, dt, rhs)
    k[1] = rhs(t, y)
    println("Hio")
    for i in 2:q
        k[i] = rhs(t + B[i, 1] * dt, y + dt * sum((B[i, j+1] * k[j] for j in 1:i-1)))
    end

    return y + dt * sum((B[end,i+1] for i in 1:q))
end

end
#f/m = dv/dt, v = dx/dt