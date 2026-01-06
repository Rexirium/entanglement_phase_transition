using MKL
using Plots, LaTeXStrings
MKL.set_num_threads(1)
include("../src/simulation.jl")

let 
    L, T = 12, 48
    b = L ÷ 2
    p, η = 0.16, 0.05   
    N = length(ARGS) == 0 ? 100 : parse(Int, ARGS[1])

    res = @time calculation_mean_multi(L, T, p, η; which_op="Sx", numsamp=N,  cutoff=eps(Float64), retstd = true, restype=Float64)
    entr = res.mean_entropy
    entr_std = res.std_entropy
    corrs = res.mean_corrs
    corrs_std = res.std_corrs
    println("Mean entropy: $entr ± $entr_std")

    plot(0:(L-1), corrs, yerror=corrs_std,
         xlabel=L"r", ylabel=L"C(r)",
         title="Correlation function at (p, η) = ($(p), $(η))",
         legend=false, framestyle=:box, lw=1)
end