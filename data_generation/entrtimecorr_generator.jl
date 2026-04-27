using Distributed
using SlurmClusterManager
using MKL
using HDF5

if Threads.nthreads() == 1
    Threads.nthreads() = 4
end
# add worker processes if none exist (use CPU-1 workers to avoid oversubscription)
if nprocs() == 1
    nworkers_to_add = max(Sys.CPU_THREADS ÷ Threads.nthreads() - 1, 1)
    addprocs(SlurmManager())
end
# make sure workers have required packages and the same MKL threading setting
@everywhere begin
    using MKL
    MKL.set_num_threads(1)
    using Statistics
    # include the entropy calculation code on all processes
    using ITensors, ITensorMPS
    ITensors.BLAS.set_num_threads(1)
    ITensors.Strided.set_num_threads(1)

    if !isdefined(Main, :RandomUnitary)
        include("../src/RandomUnitary.jl")
        using .RandomUnitary
    end
end

const nprob = 21
const ps = LinRange(0.02, 0.4, nprob)
const ns = 1 ./ ps .- 1
const L1, dL, L2 = 6, 2, 20
const Ls = L1:dL:L2

@everywhere const params = [(n, L) for L in $Ls for n in $ns]

@everywhere function calculation_wrapper(idx::Int; nsamp::Int=100)
    lsize, n = params[idx]
    corr_ops = ("Z", lsize ÷ 2, "Z", lsize ÷ 2)
    ttotal = 8lsize

    entropies = Matrix{Float64}(undef, ttotal + 1, nsamp)
    timecorrs = Matrix{Float64}(undef, lsize + 1, nsamp)
    spatcorrs = Matrix{Float64}(undef, lsize, nsamp)

    Threads.@threads for i in 1:nsamp
        ss = siteinds("S=1/2", lsize)
        psi = MPS(ComplexF64, ss, "Up")
        mnt = PMMonitor{Float64}(lsize, n)
        obs = EntropyObserver{Float64}(lsize ÷ 2)
        tcorr, _ = timecorrelation!(psi, ttotal, ttotal - lsize, mnt, corr_ops, obs; maxdim = lsize * lsize)

        entropies[:, i] = obs.entropies
        timecorrs[:, i] = tcorr
        spatcorrs[:, i] = correlation_dist(psi, "Z", "Z")
    end
    entropy_mean = mean(entropies, dims=2)[:, 1]
    entropy_sems = stdm(entropies, entropy_mean; dims=2)[:, 1] / sqrt(nsamp)
    timecorr_mean = mean(timecorrs, dims=2)[:, 1]
    timecorr_sems = stdm(timecorrs, timecorr_mean; dims=2)[:, 1] / sqrt(nsamp)
    spatcorr_mean = mean(spatcorrs, dims=2)[:, 1]
    spatcorr_sems = stdm(spatcorrs, spatcorr_mean; dims=2)[:, 1] / sqrt(nsamp)

    return entropy_mean, entropy_sems, timecorr_mean, timecorr_sems, spatcorr_mean, spatcorr_sems
end

let 
    nL = length(Ls)
    nparam = length(params)

    h5open("entr_timecorr_data_$(L1)_$(dL)_$(L2).h5", "w") do h5
        write(h5, "datatype", "Float64")
        grp = create_group(h5, "params")
        write(grp, "ps", collect(ps))  
        write(grp, "Ls", collect(Ls))
    end

    results = pmap(calculation_wrapper, 1:params)



end
