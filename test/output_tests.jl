@testset "output" begin
    # Construct generic SpectralODEProblem
    #prob = ...

    @test prepare_sampling_coverage(-1, "1 MB", prob)
    @test prepare_sampling_coverage(-1, "", prob)
    @test prepare_sampling_coverage(10, "1 MB", prob)
    @test prepare_sampling_coverage(10, "", prob)
    @test prepare_sampling_coverage(10, "-1 MB", prob) #Expect error
    # test case where N_steps/step_stride % 0 != 0
    # test case where not enough storage for two samples 
    # test cases where negative step_stride
    # test cases where step_stide = 0

    # Check that a simulation with the same name can have a new domain size
end