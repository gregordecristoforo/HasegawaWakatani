include(relpath(pwd(), @__DIR__)*"/src/HasegawaWakatini.jl")
using .Domains
using Plots
using FFTW
domain = Domain(16)
#scatter(getDomainFrequencies(domain) ./ (2 * pi))

#k_x, k_y = getDomainFrequencies(domain)
k_x = domain.kx
k_y = domain.ky

x = domain.x
for k in k_x
    display(plot!(x, sin.(k * x)))
end


domain.x[end] + domain.dx
domain.x[1]
size(domain.x)

1/(16-1)
domain.dx

domain.x[2] - domain.x[1]

## Simple test
using Plots

N = 21
xj = (0:N-1)*2*π/N
f = 2*exp.(17*im*xj) + 3*exp.(6*im*xj)# + rand(N)

original_k = 0:N-1
shifted_k = fftshift(fftfreq(N)*N)

original_fft = fft(f)
shifted_fft = fftshift(fft(f))

p1 = plot(original_k,abs.(original_fft),title="Original FFT Coefficients", xticks=original_k[1:2:end], legend=false, ylims=(0,70));
p1 = plot!([1,7,18],abs.(original_fft[[1,7,18]]),markershape=:circle,markersize=6,linecolor="white");
p2 = plot(shifted_k,abs.(shifted_fft),title="Shifted FFT Coefficients",xticks=shifted_k[1:2:end], legend=false, ylims=(0,70));
p2 = plot!([-4,0,6],abs.(shifted_fft[[7,11,17]]),markershape=:circle,markersize=6,linecolor="white");
plot(p1,p2,layout=(2,1))

##
k = 2*π/1
x = range(0,1,100)
fs = 100


N = 100
L = 2.5

xj = L*(0:N-1)/N .- L/2
freq = 2*π*fftfreq(N,N/L)

data = @. sin(freq[2]*xj)#exp.(freq[2]*im*xj)

plot(xj, real(data))
plot(freq, abs.(fft(data)))

freq[2]
abs.(fft(data))[2]

domain = Domain(N,L)
plot(domain.x, sin.(domain.kx[2].*domain.x))