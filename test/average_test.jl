using MKL
using ITensors, ITensorMPS
ITensors.BLAS.set_num_threads(1)
ITensors.Strided.set_num_threads(1)
using Statistics
using CairoMakie

if !isdefined(Main, :RandomUnitary)
    include("../src/RandomUnitary.jl")
    using .RandomUnitary
end

function entrcorr_average_wrapper(lsize::Int, ttotal::Int, param::Tuple{T,T}) where T<:Real
    dent = NHDisentangler{T}(param...)
    ss = siteinds("S=1/2", lsize)
    psi = MPS(Complex{T}, ss, "Up")
    avg = EntrCorrAverager{T}(lsize ÷ 2, lsize; n=1, op="Sz")
    # core calculation
    maxbond = 25 * lsize
    threshold = 1e-8 * (ttotal*lsize)
    truncerr = timeevolve!(psi, ttotal, dent, avg; cutoff=1e-14, maxdim=maxbond, etol=threshold)
    return avg, truncerr
end
#=
function entropy_average_wrapper(lsize::Int, ttotal::Int, param::Tuple{T,T}) where T<:Real
    dent = NHDisentangler{T}(param...)
    ss = siteinds("S=1/2", lsize)
    psi = MPS(Complex{T}, ss, "Up")
    avg = EntropyAverager{T}(lsize ÷ 2, lsize; n=1)
    # core calculation
    maxbond = 20 * lsize
    threshold = 1e-8 * (ttotal*lsize)
    truncerr = timeevolve!(psi, ttotal, dent, avg; cutoff=1e-14, maxdim=maxbond, etol=threshold)
    return avg, truncerr
end
=#

let
    L = 16
    T = 12L
    N = T - 2L
    p, η = 0.8, 0.1
    
    @timev avg, truncerr = entrcorr_average_wrapper(L, T, (p, η))

    entr_mean = avg.entr_mean
    corr_mean = avg.corr_mean
    entr_sem = sqrt(avg.entr_sstd / (N*(N-1)))
    corr_sem = sqrt.(avg.corr_sstd / (N*(N-1)))
    
    println("Entanglement Entropy at L = $L, p=$p, η=$η : $entr_mean ± $entr_sem")
    println("Truncation Error: ", truncerr)
    println("Truncation Error Threshold: ", (1e-8)*(T*L))
    
    fig = Figure()
    dist = 0 : L - 1
    ax = Axis(fig[1,1])

    lines!(ax, dist, corr_mean)
    errorbars!(ax, dist, corr_mean, corr_sem; whiskerwidth=10)
    fig
end