using FFTW
using Plots

# Parameters
D = 1.0         # Diffusion coefficient
Lx = 10.0       # Domain size in x
Ly = 10.0       # Domain size in y
Nx = 64         # Number of grid points in x
Ny = 64         # Number of grid points in y
dt = 0.01       # Time step
Nt = 100        # Number of time steps

# Create grid
x = range(0, stop=Lx, length=Nx)
y = range(0, stop=Ly, length=Ny)

# Initial condition (e.g., Gaussian)
u0 = exp.(-((x .- Lx/2).^2 .+ (y' .- Ly/2).^2))

# Precompute wave numbers for spectral method
kx = 2 * pi / Lx * [0:Nx/2; -Nx/2+1:-1]
ky = 2 * pi / Ly * [0:Ny/2; -Ny/2+1:-1]
kx2 = kx.^2
ky2 = ky.^2

# Fourier transform of the initial condition
u_hat = fft(u0)

# Function to compute the right-hand side of the spectral transformed equation
function rhs(u_hat, kx2, ky2, D)
    return -D * (kx2 .+ ky2') .* u_hat
end

# Time-stepping loop
for n = 1:Nt
    # RK3 coefficients
    alpha = [8/15, 5/12, 3/4]
    beta = [0, -17/60, -5/12]
    
    # Save the current state
    u_hat_n = u_hat
    
    # Perform RK3 steps
    for i = 1:3
        k1 = rhs(u_hat_n, kx2, ky2, D)
        u_hat = u_hat_n + dt * alpha[i] * k1
        if i < 3
            u_hat_n = beta[i] * u_hat_n + u_hat
        end
    end
    
    # Inverse Fourier transform to get back to the spatial domain
    u = ifft(u_hat)
    
    # Plotting or additional analysis can be done here
    # For example, to plot at specific time steps:
    if n % 10 == 0
        p = plot(x,y,real(u), title="Time step $(n)", xlabel="x", ylabel="y",st=:surface)
        display(p)
    end
end
