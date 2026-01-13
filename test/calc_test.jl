using MKL
using Statistics
using Plots, LaTeXStrings
MKL.set_num_threads(1)
include("../src/simulation.jl")

let 
    L, T = 12, 48
    b = L ÷ 2
    p, η = 0.5, 0.5   
    N = length(ARGS) == 0 ? 100 : parse(Int, ARGS[1])

    res = EntropyResults{Float64}(b, 1; nsamp=N)
    @timev calculation_mean_multi(L, T, p, η, res; cutoff=1e-14)
    entr = mean(res.entropies)
    entr_std = stdm(res.entropies, entr; corrected=false)
    println("Mean entropy: $entr ± $entr_std")
end