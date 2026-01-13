using MKL
using HDF5
using Distributed, SlurmClusterManager

MKL.set_num_threads(1)

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

const ps = collect(type, 0.0:0.05:1.0)
const ηs = collect(type, 0.0:0.05:1.0)
const param = vec([(p, η) for p in ps, η in ηs])

@everywhere begin
    const params = $param
    function calculation_multi_wrapper(lsize, idx)
        p, η = params[idx]
        res = EntrCorrResults{type}(lsize ÷ 2, lsize; n=1, op="Sz", nsamp=N)
        calculation_mean_multi(lsize, 4lsize, p, η, res; cutoff=cutoff)
        
        entr_mean = mean(res.entropies)
        entr_std = stdm(res.entropies, entr_mean; corrected=false)
        corr_mean = vec(mean(res.corrs, dims=2))
        corr_std = vec(stdm(res.corrs, corr_mean; corrected=false, dims=2))

        return CalcResult{type}(entr_mean, entr_std, corr_mean, corr_std)
    end
end

let 
    # Model parameters
    L1, dL, L2 = 4, 2, 18
    Ls = collect(L1:dL:L2)
    nprob, neta = length(ps), length(ηs)

    h5open("data/entr_corr_data_L$(L1)_$(dL)_$(L2)_$(nprob)x$(neta).h5", "w") do file
        write(file, "datatype", string(type))
        grp = create_group(file, "params")
        write(grp, "N", N)  
        write(grp, "ps", ps)  
        write(grp, "ηs", ηs)    
        write(grp, "Ls", Ls)
    end
    
    for L in Ls
        results = pmap(idx -> calculation_multi_wrapper(L, idx), 1:nprob*neta)

        entr_means = type[]
        entr_stds = type[]
        corr_means = Vector{type}[]
        corr_stds = Vector{type}[]

        for r in results
            push!(entr_means, r.entr_mean)
            push!(entr_stds, r.entr_std)
            push!(corr_means, r.corr_mean)
            push!(corr_stds, r.corr_std)
        end

        entr_means = reshape(entr_means, nprob, neta)
        entr_stds  = reshape(entr_stds, nprob, neta)
        corr_means = reshape(hcat(corr_means...), L, nprob, neta)
        corr_stds  = reshape(hcat(corr_stds...), L, nprob, neta)

        println("L=$L done with $N samples.")
        results = nothing  # free memory

        h5open("data/entr_corr_data_L$(L1)_$(dL)_$(L2)_$(nprob)x$(neta).h5", "r+") do file
            grpL = create_group(file, "L=$L")
            grpe = create_group(grpL, "entropy_SvN")
            write(grpe, "means", entr_means)
            write(grpe, "stds", entr_stds)

            grpc = create_group(grpL, "correlation_Sz")
            write(grpc, "means", corr_means)
            write(grpc, "stds", corr_stds)
        end
        entr_means = nothing
        entr_stds = nothing
        corr_means = nothing
        corr_stds = nothing
    end
end
