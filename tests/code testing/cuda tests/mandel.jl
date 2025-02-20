function mandel(z)
    c = z 
    maxiter = 80
    for n = 1:maxiter
        if abs(z) > 2
            return n-1
        end
        z = z^2 + c
    end
    return maxiter
end

mandel(complex(.3,-.6))

x = range(-1.5,0.5, 1000)
y = range(-1,1, 1000)

using Plots

heatmap(log.(mandel.(x' .+ im*y)))

@code_typed mandel(UInt32(1))
@code_llvm mandel(UInt32(1))
@code_native mandel(UInt32(1))