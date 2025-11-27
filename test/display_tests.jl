# ------------------------------------------------------------------------------------------
#                                      Display Tests                                        
# ------------------------------------------------------------------------------------------

using Test
using HasegawaWakatani
import HasegawaWakatani: plot_field, build_diagnostic, build_operator, get_fwd

@testset "Display Diagnostics" begin
# Minimal construction
domain = Domain(256, 256) #, MemoryType=CuArray)
ic = initial_condition(isolated_blob, domain) |> HasegawaWakatani.memory_type(domain)
dt = 0.0001

# Emulates SpectralODEProblem
prob = (; domain=domain, operators=(; solve_phi=build_operator(Val(:solve_phi), domain)), dt)

# Test 1: Test that all display diagnostics can be constructed and executed
    @testset "Display Diagnostic Construction and Execution" begin
        # Density
        display_density = build_diagnostic(Val(:plot_density); dt=dt)
        @testset "Density Display" begin
            @test display_density.name == "Plot density"
            @test display_density.metadata == "Display density"
            @test try
                display_density(ic, prob, 0.022)
                true
            catch e
                @info("Display density failed: $e")
                false
            end
        end
        
        # Vorticity
        display_vorticity = build_diagnostic(Val(:plot_vorticity); dt=dt)
        @testset "Vorticity Display" begin
            @test display_vorticity.name == "Plot vorticity"
            @test display_vorticity.metadata == "Display vorticity"
            @test try
                display_vorticity(ic, prob, 0.022)
                true
            catch e
                @info("Display vorticity failed: $e")
                false
            end
        end
        
        # Potential
        display_potential = build_diagnostic(Val(:plot_potential); dt=dt)
        ic_hat = cat(get_fwd(domain) * ic[:, :, 1], get_fwd(domain) * ic[:, :, 1]; dims=3)
        @testset "Potential Display" begin
            @test display_potential.name == "Display potential"
            @test try
                display_potential(ic_hat, prob, 0.0225)
                true
            catch e
                @info("Display potential failed: $e")
                false
            end
        end
    end
     
# Test 2: Test that the number of significant digits works
    @testset "Significant Digits Handling" for dt_test in [0.1, 0.01, 0.001, 0.00023, 0.0004567]
            digits_expected = ceil(Int, -log10(dt_test))
            display_density = build_diagnostic(Val(:plot_density); dt=dt_test)
            @test display_density.kwargs[:digits] == digits_expected
            display_vorticity = build_diagnostic(Val(:plot_vorticity); dt=dt_test)
            @test display_vorticity.kwargs[:digits] == digits_expected
            display_potential = build_diagnostic(Val(:plot_potential); dt=dt_test)
            @test display_potential.kwargs[:digits] == digits_expected
    end

end