# TODO move to test/examples folder
# --------------------------- Analytical solutions -----------------------------------------

function HeatEquationAnalyticalSolution(u0, domain, p, t)
    u0_hat = (domain.transform.FT * u0) .* exp.(p["nu"] * domain.SC.laplacian * t)
    domain.transform.iFT * u0_hat
end

function HeatEquationAnalyticalSolution2(u0, domain, p, t)
    exp.(-(domain.x' .^ 2 .+ domain.y .^ 2) / (2 * (1 + 2 * p["nu"] * t))) / (1 + 2 * p["nu"] * t)
end

# Burgers equation

# Part of "analytical" solution to Burgers equation with Gaussian waveform
function gaussian_diff_y(x, y; A=1, B=0, l=1)
    -A * y * exp(-(y^2) / (2 * l^2)) / (l^2)
end

function implicitBurgerSolution(u, x, t, f)
    u - f(x - u * t)
end

#implicitInviscidBurgerSolution.(0.1, 0, 1, y -> gaussian(0, y, l=0.08))

# No need to be part of package
# using Roots

function burgers_equation_analytical_solution(u0, domain::AbstractDomain, p, t, f=y -> gaussian(0, y, l=1))
    [find_zero.(u -> implicitBurgerSolution(u, domain.y[yi], t, f), u0[yi, xi])
     for yi in eachindex(domain.y), xi in eachindex(domain.x)]
end

# ------------------------------- Convergence testing --------------------------------------
function test_timestep_convergence(prob, analyticalSolution, timesteps, scheme=MSS3();
    physical_transform=identity, displayResults=true, kwargs...)

    #Initialize storage
    residuals = zeros(size(timesteps))

    for (i, dt) in enumerate(timesteps)
        println("Running dt = $dt")

        # Create new spectralODEProblem with new dt
        newProb = SpectralODEProblem(prob.L, prob.N, prob.domain, prob.u0, prob.tspan, p=prob.p, dt=dt)

        output = Output(newProb, 2, [], store_hdf=false, physical_transform=physical_transform)

        #Calculate approximate solution
        sol = spectral_solve(newProb, scheme, output)

        #Calculate analyticalSolution
        u = analyticalSolution(prob.u0, prob.domain, prob.p, last(sol.t); kwargs...)

        residuals[i] = norm(sol.u[end] - u)
        display(surface(domain, u))
        display(surface(domain, sol.u[end]))
    end

    if displayResults
        # Plot residuals vs. time
        display(plot(timesteps, residuals, xaxis=:log, yaxis=:log, xlabel="dt", ylabel=L"||u-u_a||"))
    end

    return timesteps, residuals
end

#
function test_resolution_convergence(prob, initialField, analyticalSolution, resolutions,
    scheme=MMS3(); displayResults=true, oneDimensional=false, physical_transform=identity, kwargs...)

    od = prob.domain
    residuals = zeros(size(resolutions))

    for (i, N) in enumerate(resolutions)

        # Create higher resolution domain
        if oneDimensional
            domain = Domain(1, N, od.Lx, od.Ly, real_transform=od.real_transform, anti_aliased=od.anti_aliased)
        else
            domain = Domain(N, N, od.Lx, od.Ly, real_transform=od.real_transform, anti_aliased=od.anti_aliased)
        end
        u0 = initial_condition(initialField, domain) #TODO rethink initial condition 

        # Create new spectralODEProblem but with updated resolution
        newProb = SpectralODEProblem(prob.L, prob.N, domain, u0, prob.tspan, p=prob.p, dt=prob.dt)
        output = Output(newProb, 2, [], store_hdf=false, physical_transform=physical_transform)

        println("Running N = $N")

        # Calculate solutions
        sol = spectral_solve(newProb, scheme, output)
        u = analyticalSolution(u0, domain, prob.p, last(sol.t); kwargs...)

        # Scaled residual to compensate for increased resolution
        residuals[i] = norm(sol.u[end] - u) / (domain.Nx * domain.Ny)
    end

    if displayResults
        display(plot(resolutions, residuals, xaxis=:log2, yaxis=:log, xlabel=L"N_x \wedge N_y",
            marker=:circle, ylabel=L"||u-u_a||/N_xN_y"))
        display(plot!(resolutions, 1e-5 * resolutions .^ -2, linestyle=:dash))
        display(plot!(resolutions, 1e-5 * resolutions .^ -1, linestyle=:dash))
    end

    return resolutions, residuals
end

#plot(burgers_equation_analytical_solution(domain, 0.155))

# ------------------------------- Old "diagnostics" ----------------------------------------
#"""
#Checks if any of the many arguments that Plots.plot is complex 
#and if so takes the real part of the inverse Fourier transform.
#"""
"""
    ifftPlot(args...; kwargs...)
    Plot the real part of the inverse Fourier transform (IFFT) of each argument that is a complex array. 
    This function is designed to handle multiple input arrays and plot them using the `plot` function 
    from a plotting library such as Plots.jl. Non-complex arrays are plotted as-is.

    # Arguments
    - `args...`: A variable number of arguments. Each argument can be an array. If the array is of a complex type, 
    its IFFT is computed, and only the real part is plotted. If the array is not complex, it is plotted directly.
    - `kwargs...`: Keyword arguments that are passed directly to the `plot` function to customize the plot.

    # Usage
    using FFTW, Plots

    # Create some sample data
    x = rand(ComplexF64, 100)\\
    y = rand(100)

    # Plot the real part of the IFFT of `x` and `y` directly
    ifftPlot(x, y, title="IFFT Plot Example", legend=:topright)

"""
function ifftPlot(args...; kwargs...)
    processed_args = []
    for arg in args
        if eltype(arg) <: Complex
            arg = real(ifft(arg))
        end
        push!(processed_args, arg)
    end

    plot(processed_args...; kwargs...)
end

function compare(x, y, A::Matrix, B::Matrix)
    println(norm(A - B))
    #plot(x, y, A)
    #plot(x,x,B)
end

# Extending PlotlyJS to easily plot surfaces when using Plots for academic figures
function plotlyjsSurface(args...; kwargs...)
    i = findfirst(k -> k === :z, keys(kwargs))
    kwargs = collect(pairs(kwargs))
    kwargs[i] = :z => transpose(kwargs[i][2])
    PlotlyJS.plot(PlotlyJS.surface(args...; kwargs...))
end

# -------------------------------- Cosmoplot related ---------------------------------------

#using InverseFunctions
# base = :(Base.Fix1(log,3))
# inverse(Base.Fix1(log,3))(1)

# function cosmo_log_formatter(x, base::Symbol=:log10)
#     println(x)
#     log_val = eval(base)(x)
#     if log_val in [0, 1]
#         return "\$$(round(Int, x))\$"
#     else
#         return "\$$(round(Int,inverse(eval(base))(1)))^{$(round(Int, log_val))}\$"
#     end
# end

# TODO implement properly to extend the loglog functionality
function get_base(scale::Symbol)
    if scale == :log
        return :e
        #What about log1p
    elseif string(scale)[1:3] == "log"
        return tryparse(Int, string(scale)[4:end])
    else

    end
end

#println(cosmo_log_formatter(100,:log10))

# Your typical matlab implementation
function loglog(x, args...; base::Symbol=:log10, kwargs...)
    plot(x, args...; xscale=base, yscale=base, xformatter=x -> cosmo_log_formatter(x, base),
        yformatter=x -> cosmo_log_formatter(x, base), kwargs...)
end

# semilogx([0.1,1],base=:log2)
function semilogx(x, args...; base::Symbol=:log10, kwargs...)
    plot(x, args...; xscale=base, xformatter=x -> cosmo_log_formatter(x, base), kwargs...)
end

function semilogy(x, args...; base::Symbol=:log10, kwargs...)
    plot(x, args...; yscale=base, yformatter=x -> cosmo_log_formatter(x, base), kwargs...)
end

# Default plot style (follows cosmoplots https://github.com/uit-cosmo/cosmoplots/blob/main/cosmoplots/default.mplstyle)
# default(frame=:box, dpi=300, size=(300 * 3.37, 300 * 2.08277), fontfamily="Computer Modern",
#     titlefontsize=8, guidefontsize=8, tickfontsize=8, legendfontsize=8, legendfontcolor=:black,
#     legendtitlefontcolor=:black, legendtitlefontsize=8, linewidth=0.75, grid=false,
#     minorticks=true, markersize=2.25, widen=1.1, aspect_ratio=:equal)

# TODO add support for log plots, yikes
#= import PythonPlot
macro pythonticks(expr)
    # Ensure the macro is hygienic
    @assert expr.head == :call "The macro only works with function calls."
    @assert expr.args[1] == :plot || expr.args[1] == :(Plots.plot) "The macro only works with `plot` or `Plots.plot`."
    # Create figure and give it right font size for the right tick spacing
    PythonPlot.figure(figsize=(3.37, 2.08277), dpi=300, num="__temporary__")
    PythonPlot.matplotlib.rcParams["font.size"] = 8
    # Remove keyword arguments from the python plot call
    args = [arg for arg in expr.args[2:end] if !(arg isa Expr && arg.head == :kw)]
    func = Expr(:call, :(PythonPlot.plot), args...)
    eval(func)
    # Add xticks and yticks based on native matplotlib ticks
    xticks = PythonPlot.PyArray(PythonPlot.gca().get_xticks())
    yticks = PythonPlot.PyArray(PythonPlot.gca().get_yticks())
    new_args = copy(expr.args)
    push!(new_args, Expr(:kw, :xticks, xticks))
    push!(new_args, Expr(:kw, :yticks, yticks))

    # Close the python figure
    PythonPlot.plotclose()

    # Construct a new call to `plot` with the injected ticks
    new_expr = Expr(:call, new_args...)
    return esc(new_expr)  # Return the modified expression
end =#