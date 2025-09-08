# ------------------------------------------- Boundary diagnostics ---------------------------------------------------------
function lowerXBoundary(u::T) where {T<:AbstractArray}
    u[1, :]
end

function upperXBoundary(u::T) where {T<:AbstractArray}
    u[end, :]
end

function lowerYBoundary(u::T) where {T<:AbstractArray}
    u[:, 1]
end

function upperYBoundary(u::T) where {T<:AbstractArray}
    u[:, end]
end

function plotBoundaries(domain::D, u::T) where {D<:AbstractDomain,T<:AbstractArray}
    lx = domain.y[1]
    ux = domain.y[end]
    ly = domain.x[1]
    uy = domain.x[end]
    labels = [@sprintf("y = %5.2f", lx) @sprintf("y = %5.2f", ux) @sprintf("x = %5.2f", ly) @sprintf("x = %5.2f", uy)]
    plot([domain.y, domain.y, domain.x, domain.x], [lowerXBoundary(u), upperXBoundary(u), lowerYBoundary(u), upperYBoundary(u)], labels=labels)
end

function maximumBoundaryValue(u::T) where {T<:AbstractArray}
    maximum([lowerXBoundary(u) upperXBoundary(u) lowerYBoundary(u) upperYBoundary(u)])
end