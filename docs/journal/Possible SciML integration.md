# SciML integration steps
The hardest issue is the spectral operators part, as they need the domain metadata to dispatch
correctly and have the operators cached. I guess one way would be to remove the cached data all 
together, however that would probably lead to slow downs. 

Assuming the `u` is a Struct that stores the `Domain`, then to integrate with SciML one would
use either a `SplitFunction` to store both `L` and `N` or a `SplitODEProblem`. As for the 
algorithm, the implemented `SBDF` family corresponds to the `MSS` family implemented in this 
code. However the `L` is calculated on each step, instead of assuming it is constant.

The outputting would work trough the use of `callback`, which probably gives enough data that
one could use a global `Output` object and with the "same" handle_output! method.

Of course it would require alot of rewritting in the code and using the correct methods and Types.