using MKL
using Statistics
using Plots, LaTeXStrings
#MKL.set_num_threads(1)

include("../src/time_evolution.jl")

MKL.set_num_threads(1)
ITensors.BLAS.set_num_threads(1)
ITensors.Strided.set_num_threads(1)

function entrcorr_average_wrapper(lsize::Int, ttotal::Int, param::Tuple{T,T}) where T<:Real
    p, η = param
    ss = siteinds("S=1/2", lsize)
    psi = MPS(Complex{T}, ss, "Up")
    avg = EntrCorrAverager{T}(lsize ÷ 2, lsize; n=1, op="Sz")
    # core calculation
    truncerr = mps_evolve!(psi, ttotal, p, η, avg; cutoff=cutoff)
    return avg, truncerr
end

let
    L, T = 12, 120
    p, η = 0.2, 0.8
    avg, truncerr = entrcorr_average_wrapper(L, T, (p, η))
    entr_std = sqrt(avg.entr_sstd / (T - 2L))
    yerror = sqrt.(avg.corr_sstd ./ (T - 2L))
    println("Entanglement Entropy Mean at p=$p, η=$η : ", avg.entr_mean)
    println("Entanglement Entropy Std at p=$p, η=$η : ", entr_std)
    println("Truncation Error Sum: ", truncerr)
    
    plot(0:L, avg.corr_mean, yerror; lw = 1.5, framestyle=:box, xlabel=L"l", ylabel="entropy", label="Entanglement Entropy Distribution")
end