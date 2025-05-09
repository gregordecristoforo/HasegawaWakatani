using FFTW
using BenchmarkTools
using LinearAlgebra #To get mul!

FFTW.set_num_threads(16)

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
# Outputs: 1.861816 seconds

@time test1(FT, iFT, A)
# Outputs: 1.876618 seconds

## Not in-place fft (ComplexF64)
FT = plan_fft(A)
iFT = plan_ifft(A)

@time test(FT, iFT, A)
# Outputs: 1.735572 seconds (6 allocations: 2.000 GiB, 5.83% gc time)

@time test1(FT, iFT, A, B)
# Outputs: 1.666759 seconds

## Not in-place fft (FloatF64)
A = real(A)
#B = real(B)
FT = plan_fft(A)
iFT = plan_ifft(A)

@time test(FT, iFT, A)
# Outputs: 2.804019 seconds (12 allocations: 4.000 GiB, 6.33% gc time

@time test1(FT, iFT, A, B)
# Outputs: no method matching

## Not in-place rfft (no in-place rfft seems to exist)
A = real(A)
FT = plan_rfft(A)
# To get right size
B = FT*A
iFT = plan_irfft(B, 8192)

@time test(FT, iFT, A)
# Outputs:  1.366431 seconds (8 allocations: 1.500 GiB, 14.60% gc time)

@time test1(FT, iFT, A, B)
# Outputs: 0.751313 seconds
# ^^ clear winner
FFTW.get_num_threads()
# For 4 and 16 threads too! It is probably close to the most optimal solution













## Number of threads test
FFTW.set_num_threads(16)
A = real(A)
FT = plan_rfft(A)
# To get right size
B = FT*A
iFT = plan_irfft(B, 8192)

@time test(FT, iFT, A)
# Outputs:  1.366431 seconds (8 allocations: 1.500 GiB, 14.60% gc time)

@time test1(FT, iFT, A, B)
# Outputs: 0.751313 seconds
# ^^ clear winner
FFTW.get_num_threads()
# For 14 or 16 threads seems to be optimal