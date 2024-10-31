export Gaussian

function Gaussian(x, y, A, B, l)
    B + A * exp(-(x^2 + y^2) / (2 * l^2))
end

function Gaussian(x, y; A=1, B=0, l=1)
    B + A * exp(-(x^2 + y^2) / (2 * l^2))
end