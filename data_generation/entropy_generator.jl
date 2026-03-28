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
    function entropy_mean_multi_wrapper(idx)
        prob, eta, lsize = params[idx]
        res = EntropyResults{type}(lsize ÷ 2, lsize; n=1, nsamp=N)
        calculation_mean_multi(lsize, 4lsize, prob, eta, res; cutoff=cutoff)

        entr_mean = mean(res.entropies)
        entr_sem = stdm(res.entropies, entr_mean) / sqrt(N)

        return (entr_mean, entr_sem)
    end
end

let 
    # Model parameters
    nL = length(Ls)
    nparam = length(params)

    h5open("data/nh_entropy_calc_L$(L1)_$(dL)_$(L2)_$(nprob)x$(neta).h5", "w") do file
        write(file, "datatype", string(type))
        grp = create_group(file, "params")
        write(grp, "N", N)  
        write(grp, "ps", collect(ps))  
        write(grp, "ηs", collect(ηs))    
        write(grp, "Ls", collect(Ls))
    end
    
    results = pmap(entropy_mean_multi_wrapper, 1:nparam)

    data_means = reshape([r[1] for r in results], nprob, neta, nL)
    data_sems  = reshape([r[2] for r in results], nprob, neta, nL)

    results = nothing  # free memory

    h5open("data/nh_entropy_calc_L$(L1)_$(dL)_$(L2)_$(nprob)x$(neta).h5", "r+") do file
        grp = create_group(file, "results")
        dset1 = create_dataset(grp, "entr_means", datatype(type), dataspace(nprob, neta, nL), 
            chunk=(nprob, neta, 1), compress=3)
        dset2 = create_dataset(grp, "entr_sems", datatype(type), dataspace(nprob, neta, nL), 
            chunk=(nprob, neta, 1), compress=3)
            
        write(dset1, data_means)
        write(dset2, data_sems)
    end

end
