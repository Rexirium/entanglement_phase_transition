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
    using Statistics
    # include the entropy calculation code on all processes
    include("../src/simulation.jl")
    ITensors.BLAS.set_num_threads(1)
    ITensors.Strided.set_num_threads(1)
end

@everywhere begin 
    const N = length(ARGS) == 0 ? 100 : parse(Int, ARGS[1])
    const type = Float64
    const cutoff = 1e-14
end
# define global constants for parameters

const nprob = 21
const neta = 20
const ps = LinRange{type}(0.0, 1.0, nprob)
const ηs = LinRange{type}(0.0, 1.0, neta)
const param = vec([(p, η) for p in ps, η in ηs])

@everywhere begin
    const params = $param
    function calculation_multi_wrapper(lsize, idx)
        p, η = params[idx]
        res = EntrCorrResults{type}(lsize ÷ 2, lsize; n=1, op="Sz", nsamp=N)
        calculation_mean_multi(lsize, 4lsize, p, η, res; cutoff=cutoff)

        entr_mean = mean(res.entropies)
        entr_sem = stdm(res.entropies, entr_mean) / sqrt(N)
        corr_mean = vec(mean(res.corrs, dims=2))
        corr_sem = vec(stdm(res.corrs, corr_mean; dims=2)) / sqrt(N)

        return CalcResult{type}(entr_mean, entr_sem, corr_mean, corr_sem)
    end
end

let 
    # Model parameters
    L1, dL, L2 = 4, 2, 18
    Ls = L1:dL:L2

    h5open("data/nh_entrcorr_calc_L$(L1)_$(dL)_$(L2)_$(nprob)x$(neta).h5", "w") do file
        write(file, "datatype", string(type))
        grp = create_group(file, "params")
        write(grp, "N", N)
        write(grp, "ps", collect(ps))
        write(grp, "ηs", collect(ηs))
        write(grp, "Ls", collect(Ls))
    end
    
    for L in Ls
        results = pmap(idx -> calculation_multi_wrapper(L, idx), 1:nprob*neta)

        entr_means = type[]
        entr_sems = type[]
        corr_means = Vector{type}[]
        corr_sems = Vector{type}[]

        for r in results
            push!(entr_means, r.entr_mean)
            push!(entr_sems, r.entr_sem)
            push!(corr_means, r.corr_mean)
            push!(corr_sems, r.corr_sem)
        end

        entr_means = reshape(entr_means, nprob, neta)
        entr_sems  = reshape(entr_sems, nprob, neta)
        corr_means = reshape(hcat(corr_means...), L, nprob, neta)
        corr_sems  = reshape(hcat(corr_sems...), L, nprob, neta)

        println("L=$L done with $N samples.")
        results = nothing  # free memory

        h5open("data/nh_entrcorr_calc_L$(L1)_$(dL)_$(L2)_$(nprob)x$(neta).h5", "r+") do file
            grpL = create_group(file, "L=$L")
            grpe = create_group(grpL, "entropies")
            write(grpe, "means", entr_means)
            write(grpe, "sems", entr_sems)

            grpc = create_group(grpL, "correlations")
            write(grpc, "means", corr_means)
            write(grpc, "sems", corr_sems)
        end
    end
end
