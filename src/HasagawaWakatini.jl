"""
Includes Domain, Operators, Parameters and Timestepper
"""
module HasagawaWakatini
export Domain, Operators, Timestepper
include("Helperfunctions.jl")
include("Operators.jl")
include("Timestepper.jl")
using .Helperfunctions#: Domain
using .Operators
using .Timestepper
#include("Tmp.jl")
end # module HasagawaWakatini