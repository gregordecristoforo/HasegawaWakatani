using Documenter, HasegawaWakatani

makedocs(
    sitename="HasegawaWakatani",
    authors="Johannes Mørkrid",
    modules=[HasegawaWakatani]
)

deploydocs(
    repo="github.com/JohannesMorkrid/HasegawaWakatani.jl.git"
)