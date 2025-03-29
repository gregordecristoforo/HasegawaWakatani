# HasagawaWakatani.jl
Main file to include, consisting of:

## domain.jl 

## diagnostics.jl 
Houses all the diagnostics of the type:
* Spectral
* Boundary
* Energy integrals
* Blob dynamics
* Local point diagnostic

Also features the **Diagnostic** type with functions that creates different types of diagnostics 
that can be added to 

## fftutilities.jl

## outputer.jl

## schemes.jl

## spectralODEProblem.jl

## spectralOperators.jl
This is the hardest one and where in-place is needed

## spectralSolve.jl
Contains **spectral_solve()**:
Main loop that creates the cache and integrates the problem. This main loop also passes
the cache to the outputer to output HDF5 files, and outputs the output container at the end.

Also will store the cache in the output file upon completion or interupts; to be able to 
resume or extend a simulation.

## utilities.jl
Mainly conserned with initial conditions, exact solutions and testing convergences, both 
resolution and timestep. Also includes ifftPlot and **logspace** which creates logspaced 
values.