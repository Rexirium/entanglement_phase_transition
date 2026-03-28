using Distributed
using SlurmClusterManager
using MKL
using HDF5

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

@everywhere begin 
    const N = length(ARGS) == 0 ? 100 : parse(Int, ARGS[1])
    const type = Float64
    const cutoff = 1e-14
end
# define global constants for parameters

const nprob = 21
const neta = 20
const L1, dL, L2 = 4, 2, 18
const ps = LinRange{type}(0.0, 1.0, nprob)
const ηs = LinRange{type}(0.0, 1.0, neta)
const Ls = L1:dL:L2
const param = [(p, η, L) for L in Ls for η in ηs for p in ps]

@everywhere begin
    const params = $param
    function calculation_multi_wrapper(idx)
        prob, eta, lsize = params[idx]
        res = EntrCorrResults{type}(lsize ÷ 2, lsize; n=1, op="Sz", nsamp=N)
        calculation_mean_multi(lsize, 4lsize, prob, eta, res; cutoff=cutoff)

        entr_mean = mean(res.entropies)
        entr_sem = stdm(res.entropies, entr_mean) / sqrt(N)
        corr_mean = vec(mean(res.corrs, dims=2))
        corr_sem = vec(stdm(res.corrs, corr_mean; dims=2)) / sqrt(N)

        return CalcResult{type}(entr_mean, entr_sem, corr_mean, corr_sem)
    end
end

let 
    # Model parameters
    nL = length(Ls)
    nparam = length(params)
    subs = CartesianIndices((nprob, neta, nL))

    h5open("data/nh_entrcorr_calc_L$(L1)_$(dL)_$(L2)_$(nprob)x$(neta).h5", "w") do file
        write(file, "datatype", string(type))
        grp = create_group(file, "params")
        write(grp, "N", N)
        write(grp, "ps", collect(ps))
        write(grp, "ηs", collect(ηs))
        write(grp, "Ls", collect(Ls))
    end
    
    results = pmap(calculation_multi_wrapper, 1:nparam)

    entr_means  = Array{type, 3}(undef, nprob, neta, nL)
    entr_sems   = Array{type, 3}(undef, nprob, neta, nL)
    corr_means  = zeros(type, L, nprob, neta, nL)
    corr_sems   = zeros(type, L, nprob, neta, nL)
    truncerrs   = Array{type, 3}(undef, nprob, neta, nL)

    @inbounds for idx in eachindex(subs)
        res = results[idx]
        sub = subs[idx]

        entr_means[sub] = res.entr_mean
        entr_sems[sub] = res.entr_sem
        corr_means[:, sub] .= res.corr_mean
        corr_sems[:, sub] .= res.corr_sem
    end

    results = nothing  # free memory

    h5open("data/nh_entrcorr_calc_L$(L1)_$(dL)_$(L2)_$(nprob)x$(neta).h5", "r+") do file
        grp = create_group(file, "results")
        dset1 = create_dataset(grp, "entr_means", datatype(type), dataspace(nprob, neta, nL), 
            chunk=(nprob, neta, 1), compress=3)
        write(dset1, entr_means)

        dset2 = create_dataset(grp, "entr_sems", datatype(type), dataspace(nprob, neta, nL), 
            chunk=(nprob, neta, 1), compress=3)
        write(dset2, entr_sems)

        dset3 = create_dataset(grp, "corr_means", datatype(type), dataspace(L2, nprob, neta, nL), 
            chunk=(L2, nprob, neta, 1), compress=3)
        write(dset3, corr_means)

        dset4 = create_dataset(grp, "corr_sems", datatype(type), dataspace(L2, nprob, neta, nL),
            chunk=(L2, nprob, neta, 1), compress=3)
        write(dset4, corr_sems)
    end
end
