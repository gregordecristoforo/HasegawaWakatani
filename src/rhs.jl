function N(du, u, d, p, t)
    n, Ω, ϕ = eachslice(u)
    g, κ = p["g"], p["kappa"]
    ϕ = d.solvePhi(Ω)
    du[1] = -d.poissonBracket(u, v) + d.dy(ϕ) - g * d.dy(n) - σₙ * d.spectral_exp(ϕ)
    du[2] = -d.poissonBracket(u, v) - g * d.dy(n) + σₒ * (1 - d.spectral_exp(ϕ))
    return du
end

function N(du, u, d, p, t)
    n, Ω, ϕ = eachslice(u)
    g, σₙ, σₒ = p["g"], p["kappa"]
    ϕ = solvePhi(Ω, d)
    du[1] = -poissonBracket(u, v, d) + dy(ϕ, d) - g * dy(n, d) - σₙ * spectral_exp(ϕ, d)
    du[2] = -poissonBracket(u, v, d) - g * dy(n, d) + σₒ * (1 - d.spectral_exp(ϕ, d))
    return du
end

# Objectives: CUDA support, easy to write rhs without thinking too much about BTS, 
# pre-allocated operator results, if calculated once during timestep then use calculation
function N(du, u, d, p, t)
    n, Ω, ϕ = eachslice(u)
    g, σₙ, σₒ = unpack(p)
    ϕ = solvePhi(Ω, d)
    du[1] = -poissonBracket(u, v, d) + dy(ϕ, d) - g * dy(n, d) - σₙ * spectral_exp(ϕ, d)
    du[2] = -poissonBracket(u, v, d) - g * dy(n, d) + σₒ * (1 - d.spectral_exp(ϕ, d))
    return du
end

function N(du, u, d, p, t)
    # Auxilary variables
    @unpack diff_y, poisson_bracket, solve_phi = d #-> references

    # Computations
    ϕ = solve_phi(Ω) #-> cache.phi
    dθ .= poisson_bracket(θ, ϕ) #-> store vx and vy once?
    dΩ .= poisson_bracket(Ω, ϕ) .- diff_y(θ) #, can then reuse vx and vy
end

function N(du, u, d, p, t)
    n, Ω, ϕ = eachslice(u)
    g, σₙ, σₒ = unpack(p)
    ϕ = solvePhi(Ω, d)
    #^^ This is good start

    #vv This is the hard part
    #du[3] is not utilized because phi is not updated!
    du[1] = -poissonBracket(u, v, d) + diff_y(ϕ, d) - g * diff_y(n, d) - σₙ * spectral_exp(ϕ, d)
    du[2] = -poissonBracket(u, v, d) - g * diff_y(n, d) + σₒ * (1 - d.spectral_exp(ϕ, d))
    return du
end