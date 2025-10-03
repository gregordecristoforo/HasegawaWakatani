#---------------------------------- CFL ----------------------------------------------------

# Calculate velocity assuming U_ExB = ̂z×∇Φ   
function vExB(u::U, domain::D) where {U<:AbstractArray,D<:AbstractDomain}
    W = u[:, :, 2] #Assumption
    W_hat = get_fwd(domain) * W
    phi_hat = solve_phi(W_hat, domain)
    get_bwd(domain) * -diff_y(phi_hat, domain), get_bwd(domain) * diff_x(phi_hat, domain)
end

#contourf(vExB([u0;;;u0], domain)[1].^2 .+ vExB([u0;;;u0], domain)[2].^2) 

#Returns max CFL
function cfl_ExB(u::T, prob::P, t::N) where {T<:AbstractArray,P<:SpectralODEProblem,N<:Number}
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

function CFLDiagnostic(N::Int=100)
    Diagnostic("ExB cfl", cfl_ExB, N, "max CFL x, max CFL y")
end

function radial_cfl_ExB(u::U, prob::P, t::T, v::V=vExB) where {U<:AbstractArray,
    P<:SpectralODEProblem,T<:Number,V<:Function}
    v_x, v_y = v(u, prob.domain)
    [maximum(v_x) * prob.dt / domain.dx, argmax(v_x)]
end

function RadialCFLDiagnostic(N::Int=100)
    Diagnostic("Radial CFL", radial_cfl_ExB, N, "max radial CFL, position")
end

function cfl_y(u::U, prob::P, t::T, v::V=vExB) where {U<:AbstractArray,P<:SpectralODEProblem,
    T<:Number,V<:Function}
    v_x, v_y = v(u, prob.domain)
    [maximum(v_y) * prob.dt / domain.dy, argmax(v_y)]
end

# TODO test Burger and maybe remove it and include in the above functions
# CFL where field is velocity
function burgerCFL(u::U, prob::P, t::T) where {U<:AbstractArray,P<:SpectralODEProblem,T<:Number}
    maximum(u) * prob.dt / prob.domain.dy
end

function BurgerCFLDiagnostic(N::Int=100)
    Diagnostic("Burger CFL", burgerCFL, N, "CFL")
end