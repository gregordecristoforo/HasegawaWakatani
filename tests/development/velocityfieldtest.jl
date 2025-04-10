prob.remove_modes(u)

# Or maybe just 
remove_modes(u)
# In spectralSolve?




write(output.simulation["cache_backup"]["last_step"], 3108)
read(output.simulation["cache_backup"]["last_step"])
#3108

#write(output_simulation)

using BenchmarkTools
data_hat = A
data = zero(spectral_transform(data_hat, domain.transform.iFT))

function no_cache(data_hat, domain)
    data = spectral_transform(data_hat, domain.transform.iFT)
    exp.(data)
end

function with_cache(data, data_hat, domain)
    spectral_transform2!(data, data_hat, domain.transform.iFT)
    data .= exp.(data)
end

@btime no_cache(data_hat, domain)
# 403.434 μs (78 allocations: 1.13 MiB)

@btime with_cache(data, data_hat, domain)
# 336.642 μs (6 allocations: 260.16 KiB)

@btime spectral_transform2!(data, data_hat, domain.transform.iFT)

function spectral_transform2!(du::DU, u::U, p::FFTW.Plan) where {DU<:AbstractArray,U<:AbstractArray}
    @assert ndims(du) == ndims(u)
    idx = ntuple(_ -> :, ndims(p))
    for i in axes(u, ndims(p)+1)
        mul!(view(du, idx..., i), p, u[idx..., i])
    end
end

p = domain.transform.iFT
idx = ntuple(_ -> :, ndims(p))
mul!(data[idx...,1], p, data_hat[idx...,1])
@views mul!(data[idx...,2], p, data_hat[idx...,2])
data

data_hat
data




function spectral_transform3!(du::DU, u::U, p::FFTW.Plan) where {DU<:Union{Tuple,Vector}, 
    U<:Union{Tuple,Vector}}
    for i in eachindex(u)
        mul!(du[i], p, u[i])
    end
end

data2_hat = [data_hat[:,:,1], data_hat[:,:,2]]
data2 = zero([data[:,:,1], data[:,:,2]])

spectral_transform3!(data2, data2_hat, domain.transform.iFT)
data2[1]









































##

E_approx = zeros(100)
E_proper = zeros(100)

for i in 1:100
    data = copy(sol.simulation["fields"][:,:,:,i]) 
    #copy(read(sol.simulation["cache_backup/u"]))

    n_hat = domain.transform.FT*data[:,:,1]
    Ω_hat = domain.transform.FT*data[:,:,2]
    ϕ_hat = solvePhi(Ω_hat, domain)

    dϕdx_hat = diffX(ϕ_hat, domain)
    dϕdy_hat = diffY(ϕ_hat, domain)
    dϕdx = domain.transform.iFT*dϕdx_hat
    dϕdy = domain.transform.iFT*dϕdy_hat

    E_kin = 1/2*(dϕdx.^2 .+ dϕdy.^2)
    heatmap(E_kin)
    heatmap(domain.transform.iFT*n_hat)

    sum(E_kin)/(domain.Lx*domain.Ly)

    E_kin_hat = abs.(dϕdx_hat).^2 .+ abs.(dϕdy_hat).^2

    sum(E_kin)/(domain.Lx*domain.Ly) - (sum(E_kin_hat[1:end,:]) - 0.5*sum(E_kin_hat[1,:]))/(domain.Nx*domain.Ny*domain.Lx*domain.Ly)

    n = domain.transform.iFT*n_hat
    ϕ = domain.transform.iFT*ϕ_hat

    ν = 1e-4
    D_Ω_hat = ν*hyper_diffusion(Ω_hat,domain)
    D_n_hat = ν*hyper_diffusion(n_hat,domain)   

    D_Ω = domain.transform.iFT*D_Ω_hat
    D_n = domain.transform.iFT*D_n_hat
    heatmap(D_Ω)
    heatmap(D_n)

    heatmap(n.*D_n)
    heatmap(ϕ.*D_Ω)

    ϕD_Ω = domain.transform.iFT*quadraticTerm(ϕ_hat, D_Ω_hat, domain)
    nD_n = domain.transform.iFT*quadraticTerm(n_hat, D_n_hat, domain)
    sum(nD_n-n.*D_n)
    sum(ϕD_Ω-ϕ.*D_Ω)

    sum(n.*D_n)
    sum(ϕ.*D_Ω)

    sum(ϕ.*D_Ω .- n.*D_n)

    Γ = domain.transform.iFT*quadraticTerm(n_hat, dϕdy_hat, domain)
    heatmap(Γ)
    sum(Γ)

    C = 0.1

    ϕ_n = domain.transform.iFT*(ϕ_hat - n_hat)

    E_total = 1/2*sum(domain.transform.iFT*quadraticTerm(dϕdx_hat, dϕdx_hat, domain) + 
    domain.transform.iFT*quadraticTerm(dϕdy_hat, dϕdy_hat, domain) + 
    domain.transform.iFT*quadraticTerm(n_hat, n_hat, domain))

    E_approx[i] = (-sum(n.*dϕdy) - C*sum((ϕ-n).^2) -sum(ϕ.*D_Ω .- n.*D_n))/E_total
    E_proper[i] = (-sum(Γ) -C*sum(ϕ_n.^2) -sum(ϕD_Ω-nD_n))/E_total

    1/2*sum(domain.transform.iFT*quadraticTerm(dϕdx_hat, dϕdx_hat, domain) + 
    domain.transform.iFT*quadraticTerm(dϕdy_hat, dϕdy_hat, domain)) - 
    (sum(E_kin_hat[1:end,:]) - 0.5*sum(E_kin_hat[1,:]))/(domain.Nx*domain.Ny)
end

plot(E_proper, label="Aliased")
plot!(E_approx, ylim=[-0.1,0.2], xlabel=L"t", ylabel=L"(\partial E/\partial t)/E", title="Energy functional "*L"(C=0.1, \nu=0.0001)", label="Anti-aliased")
savefig("figures/Energy integral C=0.1, nu=0.0001.pdf")

#plot(E_proper./E_approx)

##

using BenchmarkTools
data_hat = spectral_transform(data, domain.transform.FT)
@time potential_energy_integral(data_hat, prob, 0.0)
@time kinetic_energy_integral(data_hat, prob, 0.0)
@time resistive_dissipation_integral(data_hat, prob, 0.0)
@time potential_dissipation_integral(data_hat, prob, 0.0)
@time kinetic_dissipation_integral(data_hat, prob, 0.0)
@time viscous_dissipation_integral(data_hat, prob, 0.0)
@time energy_evolution_integral(data_hat, prob, 0.0)
@time radial_flux(data_hat, prob, 0.0)

sum(ϕD_Ω-nD_n)/(domain.Lx*domain.Ly)

sum(ϕ.*D_Ω)/(domain.Lx*domain.Ly)
sum(ϕD_Ω)/(domain.Lx*domain.Ly)


(abs.(domain.SC.Laplacian).^2-domain.SC.Laplacian.*domain.SC.Laplacian)



sum(E_kin)/(domain.Lx*domain.Ly)
(sum(E_kin_hat[1:end,:]) - 0.5*sum(E_kin_hat[1,:]))/(domain.Nx*domain.Ny*domain.Lx*domain.Ly)


data = copy(sol.simulation["fields"][:,:,:,100])
data_hat = spectral_transform(data, domain.transform.FT)

energy_evolution_integral(data_hat, prob, 0.0)/
E_approx[end]/E_total
E_proper[end]/E_total

total_energy_integral(data_hat, prob, 0.0)
E_total/(domain.Nx*domain.Ny)

ϕ_hat = solvePhi(Ω_hat, domain)
dϕdy_hat = diffY(ϕ_hat, domain)
Γ1 = domain.transform.iFT*quadraticTerm(n_hat, dϕdy_hat, domain)
Γ2 = (domain.transform.iFT*n_hat).*(domain.transform.iFT*dϕdy_hat)

sum(Γ1)-sum(Γ2)


function flux_anti_aliasing(u::U, domain::D) where {U<:AbstractArray, D<:Domain}
    n_hat = u[:,:,1]
    Ω_hat = @view u[:,:,2]
    ϕ_hat = solvePhi(Ω_hat, domain)
    dϕdy_hat = diffY(ϕ_hat, domain)
    domain.transform.iFT*quadraticTerm(n_hat, dϕdy_hat, domain)
end

function flux(u::U, domain::D) where {U<:AbstractArray, D<:Domain}
    n_hat = @view u[:,:,1]
    Ω_hat = @view u[:,:,2]
    ϕ_hat = solvePhi(Ω_hat, domain)
    dϕdy_hat = diffY(ϕ_hat, domain)
    (domain.transform.iFT*n_hat).*(domain.transform.iFT*dϕdy_hat)
end

@time flux_anti_aliasing(data_hat, domain)
#0.0039
@time flux(data_hat, domain)
#0.0011

sum(flux_anti_aliasing(data_hat, domain) - flux(data_hat, domain))
sum(flux_anti_aliasing(data_hat, domain)) 
sum(flux(data_hat, domain))

vy = zeros(256,256)
n = zeros(256,256)

function flux(u::U, domain::D, vy::V, n::V) where {U<:AbstractArray, D<:Domain, V<:AbstractArray}
    n_hat = @view u[:,:,1]
    Ω_hat = @view u[:,:,2]
    ϕ_hat = solvePhi(Ω_hat, domain)
    ϕ_hat .= diffY(ϕ_hat, domain)
    mul!(vy,domain.transform.iFT,ϕ_hat)
    mul!(n,domain.transform.iFT,n_hat)
    vy.*n #.*(domain.transform.iFT*ϕ_hat)
end

@btime flux(data_hat, domain, vy, ϕ)
# 0.001008 s

@btime flux(data_hat, domain)
# 0.005 in gc

function flux_w_tasks(u::U, prob::SOP) where {U<:AbstractArray, SOP<:SpectralODEProblem}
    n_hat = u[:,:,1]
    Ω_hat = @view u[:,:,2]
    ϕ_hat = solvePhi(Ω_hat, prob.domain)
    dϕ_hat = diffY(ϕ_hat, prob.domain)
    vy = zeros(size(prob.domain.transform.FT))
    n = similar(vy)
    task_vy = Threads.@spawn mul!(vy,prob.domain.transform.iFT,dϕ_hat)
    task_n = Threads.@spawn mul!(n,prob.domain.transform.iFT,n_hat)
    wait(task_vy)
    wait(task_n)
    @threads for i in eachindex(n)
        @inbounds vy[i] *= n[i]
    end
    return vy # This is the flux time density^^
end

using Base.Threads
@btime flux_w_tasks(data_hat, prob)
# 587.863 μs (44 allocations: 2.01 MiB)

task_n = Threads.@spawn prob.domain.transform.iFT*n_hat

task_n

function flux(u::U, domain::D, vy::V, n::V) where {U<:AbstractArray, D<:Domain, V<:AbstractArray}
    n_hat = @view u[:,:,1]
    Ω_hat = @view u[:,:,2]
    ϕ_hat = solvePhi(Ω_hat, domain)
    ϕ_hat = diffY(ϕ_hat, domain)
    (domain.transform.iFT*n_hat).*(domain.transform.iFT*ϕ_hat)
end

@btime flux(data_hat, domain, vy, ϕ)


data_hat = spectral_transform(data, domain.transform.FT)
data2_hat = copy(data_hat)

@btime flux_w_tasks(data_hat, domain, vy, ϕ)

data2_hat == data_hat

sum(Γ)/(domain.Lx*domain.Ly)
radial_flux(data_hat, prob, 0.0)


























































































using JET
using BenchmarkTools
@report_call ignored_modules=(Base,HDF5,HDF5.Drivers,HDF5.API) Output(prob, 201, diagnostics, "output/Hasegawa-Wakatini debug.h5", 
simulation_name=:parameters, store_locally=true)

@report_call Domain(256, 256, 2 * pi *26.7, 2 * pi *26.7, anti_aliased=true)
@report_opt Domain(256, 256, 2 * pi *26.7, 2 * pi *26.7, anti_aliased=true)

isconcretetype(typeof(domain))

@btime N(prob.u0_hat, domain, prob.p, 0.0)
# 12.452 ms (176 allocations: 8.58 MiB)
@btime N2(prob.u0_hat, domain, prob.p, 0.0)
# 12.702 ms (182 allocations: 9.59 MiB)

function N2(u::Array{ComplexF64, 3}, d::Domain{T}, p::Dict{String, Float64}, t::Float64) where {T}
    n = @view u[:, :, 1]  # 2D slice of u
    Ω = @view u[:, :, 2]
    
    ϕ = solvePhi(Ω, d)::Matrix{ComplexF64}
    
    dn = similar(n, ComplexF64)  # Preallocate dn
    dn .= -poissonBracket(ϕ, n, d)
    dn .-= p["kappa"] .* diffY(ϕ, d)
    dn .+= p["C"] .* ϕ
    
    dΩ = similar(Ω, ComplexF64)  # Preallocate dΩ
    dΩ .= -poissonBracket(ϕ, Ω, d)
    dΩ .+= p["C"] .* (ϕ .- n)
    
    return [dn;;; dΩ]  # Concatenate dn and dΩ
end

n = @view prob.u0_hat[:, :, 1]  # 2D slice of u
Ω = @view prob.u0_hat[:, :, 2]
typeof(domain.SC.QTPlans.iFT)

N2(prob.u0_hat, domain, prob.p, 0.0)
typeof(size(domain.transform.iFT))