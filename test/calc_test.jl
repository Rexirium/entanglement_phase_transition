using MKL
using Statistics
using ITensors, ITensorMPS
ITensors.BLAS.set_num_threads(1)
ITensors.Strided.set_num_threads(1)

if !isdefined(Main, :RandomUnitary)
    include("../src/RandomUnitary.jl")
    using .RandomUnitary
end

let 
    L, T = 12, 48
    b = L ÷ 2
    p, η = 0.5, 0.5 
    N = length(ARGS) == 0 ? 100 : parse(Int, ARGS[1])

    res = EntrCorrResults{Float64}(b, L; n=1, op="Sz", nsamp=N)
    @timev calculation_mean_multi(L, T, p, η, res; cutoff=eps(Float64))
    entr_mean = mean(res.entropies)
    entr_sem = stdm(res.entropies, entr) / sqrt(N)
    
    corr_mean = mean(res.corrs, dims=2)
    corr_sem = stdm(res.corrs, corr_mean; dims=2) / sqrt(N)
    println("Mean entropy: $entr_mean ± $entr_sem")

    fig = Figure()
    dist = 0 : L - 1
    ax = Axis(fig[1,1])

    lines!(ax, dist, corr_mean)
    errorbars!(ax, dist, corr_mean, corr_sem; whiskerwidth=10)
    fig
end