using FFTW
using Plots

x = -1:0.01:1-0.01
u = exp.(-10*x.^2)

plot(x,u)

u_hat = fft(u)
k = 2 * π * fftfreq(length(x),1/0.01)

du_hat = -im*k.*u_hat

du = ifft(du_hat)

plot(real(du_hat))
plot(x,real(du))
plot(x, real(u.*du))

v_hat = ifft(u.*du)

fft(u*du/dt) = fft(ifft(u_hat)*ifft(-im*k*u_hat))

function quadraticTerm(u, v)
    if size(u) != size(v) error("u and v must have the same size") end
    t = Tuple([-N÷4+1:N + N÷4 for N in size(u)])
    U = PaddedView(0,u,t)
    V = PaddedView(0,v,t)
    i = Tuple([1+N÷4:N + N÷4 for N in size(u)])
    fft(ifft(U).*ifft(V))[i...]
end

function quadraticTerm(u, v)
    if size(u) != size(v) error("u and v must have the same size") end
    pad_size = Tuple([3*N÷2 for N in size(u)])
    pad_pos = Tuple([N÷4+1 for N in size(u)])
    U = fftshift(PaddedView(0,fftshift(u),pad_size,pad_pos))
    V = fftshift(PaddedView(0,fftshift(v),pad_size,pad_pos))
    i = Tuple([1+N÷4:N + N÷4 for N in size(u)])
    fftshift(fftshift(fft(ifft(U).*ifft(V)))[i...])
end

v_hat = quadraticTerm(u_hat,du_hat)
plot(x,1.5*real(ifft(v_hat)))

function quadraticTerm2(u, v)
    if size(u) != size(v) error("u and v must have the same size") end
    t = Tuple([-N÷4+1:N + N÷4 for N in size(u)])
    U = u
    V = v
    i = Tuple([1+N÷4:N + N÷4 for N in size(u)])
    fft(ifft(U).*ifft(V))
end

function quadraticTerm3(u, v)
    if size(u) != size(v) error("u and v must have the same size") end
    t = Tuple([-N÷4+1:N + N÷4 for N in size(u)])
    U = ifftshift(PaddedView(0,fftshift(u),t))
    V = ifftshift(PaddedView(0,fftshift(v),t))
    i = Tuple([1+N÷4:N + N÷4 for N in size(u)])
    fft(ifft(U).*ifft(V))[i...]
end


v_hat = quadraticTerm3(u_hat, du_hat)
t = Tuple([-N÷4+1:N + N÷4 for N in size(u)])
U = ifftshift(PaddedView(0,fftshift(u_hat),t))
i = Tuple([1+N÷4:N + N÷4 for N in size(u)])
v = real(ifft(U))
plot(v)

b = PaddedView(0,fftshift(u_hat),t)
plot(real(ifft(fftshift(b))))
plot(real(ifft(u_hat)))
using DSP
v = real(DSP.conv(u_hat, du_hat))[1:200]
plot(x,v)

using BenchmarkTools

@benchmark
t = Tuple([-3/2*N:3/2*N for N in size(a)])

@benchmark test()
fft(PaddedView(0, a, t))

function test()
    t = Tuple([Integer(-N/4+1):Integer(N + N/4) for N in size(a)])
    B = fft(PaddedView(0, a, t))
    B[9:40,9:40]
end

@benchmark test2()
A[9:40,9:40] = a 
A[Integer(3/2*32):97-Integer(3/2*32),Integer(3/2*32):97-Integer(3/2*32)]
function test2()
    A = zeros(48,48)
    A[9:40,9:40] = a
    fft(A)
    A[9:40,9:40]
end

t = Tuple((Integer(-3/2*N):Integer(3/2*N) for N in size(a)))
c = ifft(PaddedView(0, a, t))


using FFTW
using PaddedViews

using FFTW
using PaddedViews

function quadraticTerm(u, v)
    # Check if sizes match
    if size(u) != size(v)
        throw(ErrorException("u and v must have the same size"))
    end

    # Determine the size of each dimension and calculate padding
    Ns = size(u)
    pad_sizes = Tuple(div(N, 2) for N in Ns)  # Padding size for each dimension
    padded_sizes = Tuple(N + 2 * pad_size for (N, pad_size) in zip(Ns, pad_sizes))
    padded_sizes = (400,)
    println(padded_sizes)

    # Apply padding to Fourier coefficients
    U_padded = Array(PaddedView(0, u, padded_sizes))
    V_padded = Array(PaddedView(0, v, padded_sizes))

    # Perform IFFT on padded data
    u_real = real(ifft(U_padded))
    v_real = real(ifft(V_padded))

    # Multiply in real space
    product_real = u_real .* v_real

    # Perform FFT to return to Fourier space
    product_fourier = fft(product_real)

    # Extract the original Fourier coefficient size from the result
    original_indices = Tuple(1:N for N in Ns)
    result = product_fourier[original_indices...]

    return result
end

v_hat = quadraticTerm(u_hat, du_hat)
plot(4*real(ifft(v_hat)))

plot(real(ifft(u_hat).*ifft(du_hat)))

U = Array(PaddedView(0, u_hat, (400,)))
V = Array(PaddedView(0, v_hat, (400,)))
plot(4*real(ifft(U)).*real(ifft(V)))