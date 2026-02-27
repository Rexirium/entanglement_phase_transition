using MKL
using Statistics
using Plots, LaTeXStrings
MKL.set_num_threads(1)
include("../src/simulation.jl")
ITensors.BLAS.set_num_threads(1)
ITensors.Strided.set_num_threads(1)

let 
    L, T = 12, 48
    b = L ÷ 2
    p, η = 0.5, 0.5 
    N = length(ARGS) == 0 ? 100 : parse(Int, ARGS[1])

    res = EntrCorrResults{Float64}(b, L; n=1, op="Sz", nsamp=N)
    @timev calculation_mean_multi(L, T, p, η, res; cutoff=eps(Float64))
    entr = mean(res.entropies)
    entr_std = stdm(res.entropies, entr) / sqrt(N)
    
    corr = mean(res.corrs, dims=2)
    corr_std = stdm(res.corrs, corr; dims=2) / sqrt(N)
    println("Mean entropy: $entr ± $entr_std")

    dist = 0:(L-1)
    plot(dist, corr[:], yerror=corr_std[:], xlabel=L"r", ylabel=L"C(r)", title="Mean Correlation Function", legend=false)
end