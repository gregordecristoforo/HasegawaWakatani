using BenchmarkTools

function make_zero(up::Matrix{ComplexF64})
    up .= zero(up)
end

function make_zero2(up::Matrix{ComplexF64})
    up = zero(up)
end

function make_zero3(up::Matrix{ComplexF64})
    up .= zero.(up)
end

function make_zero4(up::Matrix{ComplexF64})
    @. up = zero(up)
end

function make_zero5(up::Matrix{ComplexF64})
    @. up = 0
end

function make_zero6(up::Matrix{ComplexF64})
    up .= 0
end

function make_zero7(up::Matrix{ComplexF64})
    up = zero.(up)
end

function make_zero8(up::Matrix{ComplexF64})
    up = zeros(eltype(up), size(up))
end

up = rand(ComplexF64, 1024, 1024)
@btime make_zero($up)
#2.718 ms

up = rand(ComplexF64, 1024, 1024)
@btime make_zero2($up)
#1.137 ms

up = rand(ComplexF64, 1024, 1024)
@btime make_zero3($up)
#819.810 μs

up = rand(ComplexF64, 1024, 1024)
@btime make_zero4($up)
#825.172 μs

up = rand(ComplexF64, 1024, 1024)
@btime make_zero5($up)
#833.161 μs

up = rand(ComplexF64, 1024, 1024)
@btime make_zero6($up)
#831.131 μs

up = rand(ComplexF64, 1024, 1024)
@btime make_zero7($up)
#1.175 ms