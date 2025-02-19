using FFTW
using BenchmarkTools
using LinearAlgebra #To get mul!

A = Matrix{ComplexF64}(rand(8192, 8192))
B = similar(A)

A[1]
Base.summarysize(A) 

function test(FT, iFT, A)#FFTW.cFFTWPlan
    iFT*A
    FT*A
    return 0
end    

function test(FT::FFTW.rFFTWPlan, iFT, A)
    iFT*(FT*A)
    return 0
end

function test1(FT, iFT, A)
    mul!(A,FT,A)
    mul!(A,iFT,A)
    return 0
end

function test1(FT, iFT, A, B)
    mul!(B,FT,A)
    mul!(A,iFT,B)
    return 0
end

## In-place test
FT = plan_fft!(A)
iFT = plan_ifft!(A)

@time test(FT, iFT, A)
# Outputs: 2.454781 seconds

@time test1(FT, iFT, A)
# Outputs: 2.454781 seconds

## Not in-place fft (ComplexF64)
FT = plan_fft(A)
iFT = plan_ifft(A)

@time test(FT, iFT, A)
# Outputs: 3.693313 seconds (6 allocations: 2.000 GiB, 6.03% gc time)

@time test1(FT, iFT, A, B)
# Outputs: 2.729167 seconds

## Not in-place fft (FloatF64)
A = real(A)
#B = real(B)
FT = plan_fft(A)
iFT = plan_ifft(A)

@time test(FT, iFT, A)
# Outputs: 4.581446 seconds (12 allocations: 4.000 GiB, 6.82% gc time)

@time test1(FT, iFT, A, B)
# Outputs: no method matching

## Not in-place rfft (no in-place rfft seems to exist)
A = real(A)
FT = plan_rfft(A)
# To get right size
B = FT*A
iFT = plan_irfft(B, 8192)

@time test(FT, iFT, A)
# Outputs: 2.899847 seconds (8 allocations: 1.500 GiB, 6.91% gc time)

@time test1(FT, iFT, A, B)
# Outputs: 2.088340 seconds
# ^^ clear winner
FFTW.get_num_threads()
# For single thread that is!