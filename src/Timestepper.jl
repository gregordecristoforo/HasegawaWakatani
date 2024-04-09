# Numerical integration method

module Timestepper
export timeStep

function timeStep(data, dt, jacobian)
    data + dt * jacobian
end
end