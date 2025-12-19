using MKL
using Plots, LaTeXStrings
MKL.set_num_threads(1)
include("../src/simulation.jl")

let 
    L, T = 10, 40
    b = L ÷ 2
    p, η = 0.9, 0.0
    N = length(ARGS) == 0 ? 100 : parse(Int, ARGS[1])

    res = @timev calculation_mean_multi(L, T, p, η; which_op="Sx", numsamp=N,  cutoff=1e-14, retstd = true, restype=Float64)
    corrs = res.mean_corrs
    corrs_std = res.std_corrs
    println(corrs)

    plot(0:(L-1), corrs, yerror=corrs_std,
         xlabel=L"r", ylabel=L"C(r)",
         title="Correlation function at (p, η) = ($(p), $(η))",
         legend=false, framestyle=:box, lw=1)
end