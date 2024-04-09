using Calculus
using .Operators
using Plots
using FFTW

# Read in input file
for line in readlines("input.txt")
    if line != ""
        if first(line) != "#" 
            println(line)
        end
    end
end

function test(a, b)
    a + b
end

derivative(sin, π / 2)

Operators.test2(2)
#Hi
N = 64
n = zeros(N, N)
x = LinRange(-4, 4, 40);
y = x;

function gaussianField(x, y, sx=1, sy=1)
    1 / (2 * π * sqrt(sx * sy)) * exp(-(x .^ 2 / sx + y .^ 2 / sy) / 2)
end

n = gaussianField.(x, y', 1, 2)

#plot!(x,y,n,st=:contourf)
#contour!(x,y,n)
plot(x, y, n)


#Sum a_jk*e^(im*j*x)*e^(im*k*y)

#b_jk = -a_jk*(j^2 + k^2)
f = fft(n)

a = fftfreq(40)

contour(x, y, real(f))

f = [-f[j, k] * (a[j]^2 + a[k]^2) for j in eachindex(x), k in eachindex(y)]

t = real(ifft!(f))

plot(x, y, t)

hdf5