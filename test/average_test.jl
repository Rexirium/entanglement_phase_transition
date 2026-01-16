using MKL
using Statistics
using Plots, LaTeXStrings
#MKL.set_num_threads(1)

include("../src/time_evolution.jl")

MKL.set_num_threads(1)
ITensors.BLAS.set_num_threads(1)
ITensors.Strided.set_num_threads(1)

function entrcorr_average_wrapper(lsize::Int, ttotal::Int, param::Tuple{type,type}) where type<:Real
    p, η = param
    ss = siteinds("S=1/2", lsize)
    psi = MPS(Complex{type}, ss, "Up")
    avg = EntrCorrAverager{type}(lsize ÷ 2, lsize; n=1, op="Sx")
    # core calculation
    mps_evolve!(psi, ttotal, p, η, avg; cutoff=eps(Float64), maxdim=512)
    return avg
end

let
    L, T = 100, 1000
    p, η = 0.8, 0.1
    @timev avg = entrcorr_average_wrapper(L, T, (p, η))

    entr_std = sqrt(avg.entr_sstd / (T - 2L))
    corr_std = sqrt.(avg.corr_sstd ./ (T - 2L))
    
    println("Entanglement Entropy Mean at p=$p, η=$η : ", avg.entr_mean)
    println("Entanglement Entropy Std at p=$p, η=$η : ", entr_std)
    println("Truncation Error Sum: ", avg.truncerr)

    plot(0:(L-1), avg.corr_mean, yerror=corr_std, lw = 1.5, framestyle=:box, xlabel=L"l", ylabel="entropy", label="Entanglement Entropy Distribution")
end