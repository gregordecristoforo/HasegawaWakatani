#---------------------------- Display diagnostic -------------------------------------------

function plot_field(domain, field, time=-1; field_name="", digits=2, kwargs...)
    display(heatmap(domain, field; aspect_ratio=:equal, xlabel="x", ylabel="y",
                    title=field_name * "(x, y)" *
                          time == -1 ? "" : "t = $(round(time, digits=digits)))",
                    kwargs...))
end

# -------------------------------------- Density -------------------------------------------

function plot_density(state, prob, time; digits=2, kwargs...)
    n = selectdim(state, ndims(state), 1) |> Array
    plot_field(prob.domain, n, time; field_name=L"n", digits=digits, kwargs...)
end

function build_diagnostic(::Val{:plot_density}; stride::Int, kwargs...)
    digits = ceil(Int, -log10(dt))
    Diagnostic("Plot density", plot_density, N, "Display density"; stores_data=false)
end

# ------------------------------------ Vorticity -------------------------------------------

function plot_vorticity(u::U, prob::P,
                        t::T) where {U<:AbstractArray,P<:SpectralODEProblem,T<:Number}
    Ω = selectdim(state, ndims(state), 2) |> Array
    plot_field(prob.domain, Ω, time; field_name=L"\Omega",
               digits=digits, color=:jet, kwargs...)
end

function build_diagnostic(::Val{:plot_density}; stride::Int, kwargs...)
    digits = ceil(Int, -log10(dt))
    Diagnostic("Plot vorticity", plot_vorticity, N, "Display vorticity"; stores_data=false)
end

# ----------------------------------- Potential --------------------------------------------

function plot_potential(state, prob, time; kwargs...)
    @unpack operators, domain, dt = prob
    @unpack solve_phi = operators
    Ω = selectdim(state, ndims(state), 2)
    ϕ = bwd(domain) * solve_phi(Ω) |> Array
    plot_field(domain, ϕ, time; field_name=L"\phi", digits=digits, kwargs...)
end

function build_diagnostic(::Val{:display_potential}, initial_data, prob, t0; stride::Int,
                          kwargs...)
    kwargs = (; digits=ceil(Int, -log10(prob.dt)))
    Diagnostic(initial_data; name="Display potential", method=plot_potential, stride=stride,
               metadata="Displays the potential", assumes_spectral_field=true,
               stores_data=false, kwargs=kwargs)
end