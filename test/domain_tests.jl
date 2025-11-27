@testset "output" begin

    Domain(256, 256, Lx=1, Ly=1) # Uses DEFAULT_OPERATORS
    SquareDomain(256, L=1) # Should be same as Domain

    # Documentation should explain operators

    # Should have operators._domain perhaps, incase user wants to use domain info in rhs

end

# check if realtransforms havles the space
# check that dealiased is true
