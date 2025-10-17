using HasegawaWakatani, Documenter
DocMeta.setdocmeta!(HasegawaWakatani, :DocTestSetup, :(using HasegawaWakatani);
                    recursive = true)

makedocs(; sitename = "HasegawaWakatani",
         authors = "Johannes Mørkrid",
         modules = [HasegawaWakatani],
         warnonly = [:doctest, :missing_docs])

deploydocs(; repo = "github.com/JohannesMorkrid/HasegawaWakatani.jl.git")