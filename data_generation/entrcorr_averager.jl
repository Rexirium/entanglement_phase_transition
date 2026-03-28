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
    const type = Float64
    const cutoff = eps(type)
end
# define global constants for parameters

const nprob = 21
const neta = 21
const L1, dL, L2 = 10, 2, 40
const ps = LinRange{type}(0.0, 1.0, nprob)
const ηs = LinRange{type}(0.0, 1.0, neta)
const Ls = L1:dL:L2
const param = [(p, η, L) for L in Ls for η in ηs for p in ps]

@everywhere begin
    const params = $param

    function entrcorr_average_wrapper(idx::Int)
        prob, eta, lsize = params[idx]
        ttotal = 12lsize
        dent = NHDisentangler{type}(prob, eta)
        ss = siteinds("S=1/2", lsize)
        psi = MPS(Complex{type}, ss, "Up")
        avg = EntrCorrAverager{type}(lsize ÷ 2, lsize; n=1, op="Sz")
        # core calculation
        threshold = 5e-7 * (ttotal * lsize)
        maxbond = 20*lsize
        truncerr = mps_evolve!(psi, ttotal, dent, avg; cutoff=cutoff, maxdim=maxbond, etol=threshold)

        psi = nothing  # free memory
        return avg, truncerr
    end
end

let 
    # Model parameters
    nL = length(Ls)
    nparam = length(params)
    subs = CartesianIndices((nprob, neta, nL))

    h5open("data/nh_entrcorr_avg_L$(L1)_$(dL)_$(L2)_$(nprob)x$(neta)_new.h5", "w") do file
        write(file, "datatype", string(type))
        grp = create_group(file, "params")
        write(grp, "ps", collect(ps))
        write(grp, "ηs", collect(ηs))
        write(grp, "Ls", collect(Ls))
    end

    entr_means  = Array{type, 3}(undef, nprob, neta, nL)
    entr_sems   = Array{type, 3}(undef, nprob, neta, nL)
    corr_means  = zeros(type, L2, nprob, neta, nL)
    corr_sems   = zeros(type, L2, nprob, neta, nL)
    truncerrs   = Array{type, 3}(undef, nprob, neta, nL)

    averagers = pmap(entrcorr_average_wrapper, 1:nparam)

    @inbounds for idx in eachindex(1:nparam)
        avg, truncerr = averagers[idx]
        sub = subs[idx]
        L = Ls[sub[3]]
        N = 10 * L

        if avg.accept
            entr_means[sub] = avg.entr_mean
            entr_sems[sub] = sqrt(avg.entr_sstd / (N*(N-1)))
            corr_means[1:L, sub] .= avg.corr_mean
            corr_sems[1:L, sub] .= sqrt.(avg.corr_sstd / (N*(N-1)))
            truncerrs[sub] = truncerr
        else
            entr_means[sub] = NaN
            entr_sems[sub] = NaN
            corr_means[1:L, sub] .= NaN
            corr_sems[1:L, sub] .= NaN
            truncerrs[sub] = NaN
        end
    end

    averagers = nothing  # free memory

    h5open("data/nh_entrcorr_avg_L$(L1)_$(dL)_$(L2)_$(nprob)x$(neta)_new.h5", "r+") do file
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

        dset5 = create_dataset(grp, "truncerrs", datatype(type), dataspace(nprob, neta, nL),
            chunk=(nprob, neta, 1), compress=3)
        write(dset5, truncerrs)
    end
  
end