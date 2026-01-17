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
    truncerr = mps_evolve!(psi, ttotal, p, η, avg; cutoff=eps(Float64), maxdim=lsize^2)
    return avg, truncerr
end

let
    L, T = 24, 160
    p, η = 0.9, 0.1
    @timev avg, truncerr = entrcorr_average_wrapper(L, T, (p, η))

    entr_std = sqrt(avg.entr_sstd / (T - 2L))
    corr_std = sqrt.(avg.corr_sstd ./ (T - 2L))
    
    println("Entanglement Entropy at p=$p, η=$η : $(avg.entr_mean) ± $entr_std")
    println("Truncation Error: ", truncerr)
    println("Truncation Error Floor: ", eps(Float64)*(T*L/2))

    plot(0:(L-1), avg.corr_mean, yerror=corr_std;
        lw = 1.5, framestyle=:box, xlabel=L"r", ylabel=L"C(r)", label="Correlation Function")
end