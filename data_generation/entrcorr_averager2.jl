using MKL
using HDF5
using Distributed, SlurmClusterManager

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
    include("../src/time_evolution.jl")
    ITensors.BLAS.set_num_threads(1)
    ITensors.Strided.set_num_threads(1)
end

@everywhere begin
    const type = Float64
    const cutoff = eps(type)
    const η0 = type(0.05)
end
# define global constants for parameters
const L1, dL, L2 = 10, 2, 40
const nprob = 51
const ps = LinRange{type}(0.0, 1.0, nprob)
const Ls = L1:dL:L2

@everywhere begin

    function entrcorr_average_wrapper(prob::Real, lsize::Int)
        ttotal = 12lsize
        dent = NHCNOTDisentangler{type}(prob, η0)

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
    nL = length(Ls)
    params = [(p, L) for L in Ls for p in ps]
    nparam = length(params)
    subs = CartesianIndices((nprob, nL))

    h5open("data/nhcnot_entrcorr_avg_L$(L1)_$(dL)_$(L2)_$(nprob)x1.h5", "w") do file
        write(file, "datatype", string(type))
        grp = create_group(file, "params")
        write(grp, "ps", collect(ps))
        write(grp, "ηs", [η0])
        write(grp, "Ls", collect(Ls))
    end

    averagers = pmap(pl -> entrcorr_average_wrapper(pl...), params)

    entr_means = Matrix{type}(undef, nprob, nL)
    entr_sems = Matrix{type}(undef, nprob, nL)
    corr_means = zeros(type, L2, nprob, nL)
    corr_sems = zeros(type, L2, nprob, nL)
    truncerrs = Vector{type}(undef, nprob)

    @inbounds for idx in eachindex(averagers)
        avg, truncerr = averagers[idx]
        sub = subs[idx]
        L = Ls[sub[2]]
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

    h5open("data/nhcnot_entrcorr_avg_L$(L1)_$(dL)_$(L2)_$(nprob)x1.h5", "r+") do file
        grp = create_group(file, "results")
        dset1 = create_dataset(grp, "entr_means", datatype(type), dataspace(nprob,  nL))
        write(dset1, entr_means)

        dset2 = create_dataset(grp, "entr_sems", datatype(type), dataspace(nprob, nL))
        write(dset2, entr_sems)

        dset3 = create_dataset(grp, "corr_means", datatype(type), dataspace(L2, nprob, nL))
        write(dset3, corr_means)

        dset4 = create_dataset(grp, "corr_sems", datatype(type), dataspace(L2, nprob, nL))
        write(dset4, corr_sems)

        dset5 = create_dataset(grp, "truncerrs", datatype(type), dataspace(nprob, nL))
        write(dset5, truncerrs)
    end
end