using PaddedViews

function quadraticTerm(u, v)
    if size(u) != size(v)
        error("u and v must have the same size")
    end
    t = Tuple([-N÷4+1:N+N÷4 for N in size(u)])
    U = ifftshift(PaddedView(0, fftshift(u), t)[t...])
    V = ifftshift(PaddedView(0, fftshift(v), t)[t...])
    i = Tuple([1+N÷4:N+N÷4 for N in size(u)])
    ifftshift(fftshift(fft(ifft(U) .* ifft(V)))[i...])
end

function quadraticTerm2(u, v)
    if size(u) != size(v)
        error("u and v must have the same size")
    end
    fft(ifft(u) .* ifft(v))
end