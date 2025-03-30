# ------------------------------------------- Boundary diagnostics ---------------------------------------------------------
function lowerXBoundary(u::Array)
    u[1, :]
end

function upperXBoundary(u::Array)
    u[end, :]
end

function lowerYBoundary(u::Array)
    u[:, 1]
end

function upperYBoundary(u::Array)
    u[:, end]
end

function plotBoundaries(domain::Domain, u::Array)
    lx = domain.y[1]
    ux = domain.y[end]
    ly = domain.x[1]
    uy = domain.x[end]
    labels = [@sprintf("y = %5.2f", lx) @sprintf("y = %5.2f", ux) @sprintf("x = %5.2f", ly) @sprintf("x = %5.2f", uy)]
    plot([domain.y, domain.y, domain.x, domain.x], [lowerXBoundary(u), upperXBoundary(u), lowerYBoundary(u), upperYBoundary(u)], labels=labels)
end

function maximumBoundaryValue(u::Array)
    maximum([lowerXBoundary(u) upperXBoundary(u) lowerYBoundary(u) upperYBoundary(u)])
end