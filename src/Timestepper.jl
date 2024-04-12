# Numerical integration method

module Timestepper
export timeStep

function timeStep(y, dt, jacobian)
    return y + dt*jacobian(y,t)
end

end

#f/m = dv/dt, v = dx/dt