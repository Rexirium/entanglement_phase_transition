using MKL
using ITensors, ITensorMPS
using Statistics
ITensors.BLAS.set_num_threads(1)
ITensors.Strided.set_num_threads(1)
if !isdefined(Main, :RandomUnitary)
    include("../src/RandomUnitary.jl")
    using .RandomUnitary
end

function calculation_wrapper(lsize::Int, n::Real; nsamp::Int=100)
    corr_ops = ("Z", lsize ÷ 2, "Z", lsize ÷ 2)
    entropies = Matrix{Float64}(undef, 4lsize + 1, nsamp)
    timecorrs = Matrix{Float64}(undef, 2lsize + 1, nsamp)
    Threads.@threads for i in 1:nsamp
        ss = siteinds("S=1/2", lsize)
        psi = MPS(ComplexF64, ss, "Up")
        mnt = PMMonitor{Float64}(lsize, n)
        obs = EntropyObserver{Float64}(lsize ÷ 2)
        tcorr, _ = timecorrelation!(psi, 4lsize, 2lsize, mnt, corr_ops, obs; maxdim = lsize * lsize)

        entropies[:, i] = obs.entropies
        timecorrs[:, i] = tcorr
    end
    entropy_mean = mean(entropies, dims=2)
    entropy_sems = stdm(entropies, entropy_mean; dims=2) / sqrt(nsamp)
    timecorr_mean = mean(timecorrs, dims=2)
    timecorr_sems = stdm(timecorrs, timecorr_mean; dims=2) / sqrt(nsamp)

    return entropy_mean[:, 1], entropy_sems[:, 1], timecorr_mean[:, 1], timecorr_sems[:, 1]
end

let 
    L = 12
    n = 7
    @time res = calculation_wrapper(L, n)
    @show res[3]
end