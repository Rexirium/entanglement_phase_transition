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
    avg = EntrCorrAverager{T}(lsize ÷ 2, lsize; n=1, op="Sx")
    # core calculation
    truncerr = mps_evolve!(psi, ttotal, p, η, avg; cutoff=1e-14, maxdim=10*lsize)
    return avg, truncerr
end

function entropy_average_wrapper(lsize::Int, ttotal::Int, param::Tuple{T,T}) where T<:Real
    p, η = param
    ss = siteinds("S=1/2", lsize)
    psi = MPS(Complex{T}, ss, "Up")
    avg = EntropyAverager{T}(lsize ÷ 2, lsize; n=1)
    # core calculation
    truncerr = mps_evolve!(psi, ttotal, p, η, avg; cutoff=1e-14, maxdim=10*lsize)
    entr_std = sqrt(avg.entr_sstd / (ttotal - 2lsize))
    return avg.entr_mean, entr_std, truncerr
end

let
    L = 32
    T = 12L
    p, η = 0.7, 0.2
    
    @timev avg, truncerr = entrcorr_average_wrapper(L, T, (p, η))

    entr_mean = avg.entr_mean
    corr_mean = avg.corr_mean
    entr_std = sqrt(avg.entr_sstd / (T - 2L))
    corr_std = sqrt.(avg.corr_sstd ./ (T - 2L))
    
    println("Entanglement Entropy at L = $L, p=$p, η=$η : $entr_mean ± $entr_std")
    println("Truncation Error: ", truncerr)
    println("Truncation Error Ceiling: ", (1e-14)*(T*L/2))
    
    plot(0:(L-1), corr_mean, yerror=corr_std;
        lw = 1.5, framestyle=:box, xlabel=L"r", ylabel=L"C(r)", label="Correlation Function", 
        title=latexstring("L=$L, p=$p, η=$η"))
    
end