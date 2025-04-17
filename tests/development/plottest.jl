using Plots

default(frame=:box, dpi=300, size=(100 * 3.37, 100 * 2.08277), fontfamily="Computer Modern",
    guidefontsize=8, tickfontsize=8, legendfontsize=8, legendfontcolor=:black, legendtitlefontcolor=:black,
    legendtitlefontsize=8, linewidth=0.75, grid=false, minorticks=true, markersize=2.25, widen=1.1)

#using PythonPlot
pythonplot(ticks=:native, dpi=300, fontfamily="Computer Modern")
# Axes and tick settings
default(#axeslinewidth=0.5
    xticks=:both,
    yticks=:both)
gr()

function cosmoplot(x, args...; kwargs...)
    PythonPlot.figure(figsize=(3.37, 2.08277), dpi=300)
    PythonPlot.matplotlib.rcParams["font.size"] = 8
    PythonPlot.plot(x)
    display(plot(x, args...; xticks=PythonPlot.PyArray(PythonPlot.gca().get_xticks()),
        yticks=PythonPlot.PyArray(PythonPlot.gca().get_yticks()), kwargs...))
    PythonPlot.plotclose()
end

cosmoplot(data, xlabel="X axis", ylabel="Y axis")
plot(data, xlabel="X axis", ylabel="Y axis")

import PythonPlot
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
end

x = exp.(LinRange(-3, 5, 100))
@pythonticks plot(x, x, xlabel="X axis", ylabel="Y axis", label="y1")
plot(x, x, xlabel="X axis", ylabel="Y axis", label="y1")

data = exp.(LinRange(-3, 5, 100))
@pythonticks plot(data, label=false, xlabel="X axis", ylabel="Y axis")
plot(data, label=false, xlabel="X axis", ylabel="Y axis")
