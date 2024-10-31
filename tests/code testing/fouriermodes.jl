m11 = @. u_hat[1,1]*exp(im*domain.kx[1]*domain.x)*exp(im*domain.ky[1]*domain.y')
plot(domain.x, domain.y, real(m11),st=:surface)
m21 = @. u_hat[2,1]*exp(im*domain.kx[2]*domain.x)*exp(im*domain.ky[1]*domain.y')
plot(domain.x, domain.y, real(m21),st=:surface)
m31 = @. u_hat[3,1]*exp(im*domain.kx[3]*domain.x)*exp(im*domain.ky[1]*domain.y')
plot(domain.x, domain.y, real(m31),st=:surface)
m41 = @. u_hat[4,1]*exp(im*domain.kx[4]*domain.x)*exp(im*domain.ky[1]*domain.y')
plot(domain.x, domain.y, real(m41),st=:surface)
m12 = @. u_hat[1,2]*exp(im*domain.kx[1]*domain.x)*exp(im*domain.ky[2]*domain.y')
plot(domain.x, domain.y, real(m12),st=:surface)
m13 = @. u_hat[1,3]*exp(im*domain.kx[1]*domain.x)*exp(im*domain.ky[3]*domain.y')
plot(domain.x, domain.y, real(m13),st=:surface)

domain = Domain(4)
for kx in domain.kx
    for ky in domain.ky
        m = @. exp(im*kx*domain.x')*exp(im*ky*domain.y)
        plot(domain.x, domain.y, real(m),st=:heatmap)
        title!("kx:"*string(kx/(2*pi))*", ky:"*string(ky/(2*pi)))
        xlabel!("x")
        display(ylabel!("y"))
    end
end

u_hat
plot(domain.x, domain.y, real(irfft(u_hat, 4)), st=:surface)
xlabel!("x")
