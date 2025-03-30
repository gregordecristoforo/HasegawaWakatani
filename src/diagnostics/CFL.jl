#---------------------------------- CFL ----------------------------------------------------

# Calculate velocity assuming U_ExB = ̂z×∇Φ   
function vExB(u::AbstractArray, domain::Domain)
    W = u[:, :, 2] #Assumption
    W_hat = domain.transform.FT * W
    phi_hat = solvePhi(W_hat, domain)
    domain.transform.iFT * -diffY(phi_hat, domain), domain.transform.iFT * diffX(phi_hat, domain)
end

#contourf(vExB([u0;;;u0], domain)[1].^2 .+ vExB([u0;;;u0], domain)[2].^2) 

#Returns max CFL
function cfl_ExB(u::AbstractArray, prob::SpectralODEProblem, t::Number)
    v_x, v_y = vExB(u, prob.domain)
    #(CFLx, CFLy, x, y)
    # if maximum(v_x) * prob.dt / prob.domain.dx >= 0.5
    #     println("Breakdown t=$t")
    # elseif maximum(v_y) * prob.dt / prob.domain.dx >= 0.5
    #     println("Breakdown t=$t")
    # end
    println("CFL", maximum(v_x) * prob.dt / prob.domain.dx)
    [maximum(v_x) * prob.dt / prob.domain.dx, maximum(v_y) * prob.dt / prob.domain.dy]
end

function CFLDiagnostic(N=100)
    Diagnostic("ExB cfl", cfl_ExB, N, "max CFL x, max CFL y")
end

function radial_cfl_ExB(u::AbstractArray, prob::SpectralODEProblem, t::Number, v::Function=vExB)
    v_x, v_y = v(u, prob.domain)
    [maximum(v_x) * prob.dt / domain.dx, argmax(v_x)]
end

function RadialCFLDiagnostic(N=100)
    Diagnostic("Radial CFL", radial_cfl_ExB, N, "max radial CFL, position")
end

function cfl_y(u::AbstractArray, prob::SpectralODEProblem, t::Number, v::Function=vExB)
    v_x, v_y = v(u, prob.domain)
    [maximum(v_y) * prob.dt / domain.dy, argmax(v_y)]
end

# TODO test Burger and maybe remove it and include in the above functions
# CFL where field is velocity
function burgerCFL(u::AbstractArray, prob::SpectralODEProblem, t::Number)
    maximum(u) * prob.dt / prob.domain.dy
end

function BurgerCFLDiagnostic(N=100)
    Diagnostic("Burger CFL", burgerCFL, N, "CFL")
end