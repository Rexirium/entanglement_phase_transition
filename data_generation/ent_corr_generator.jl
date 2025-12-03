using MKL
using HDF5
using Distributed, SlurmClusterManager

MKL.set_num_threads(1)

# add worker processes if none exist (use CPU-1 workers to avoid oversubscription)
if nprocs() == 1
    nworkers_to_add = max(Sys.CPU_THREADS ÷ Threads.nthreads() - 2, 1)
    addprocs(SlurmManager(), nworkers_to_add)
end
# make sure workers have required packages and the same MKL threading setting
@everywhere using MKL
@everywhere MKL.set_num_threads(1)

# include the entropy calculation code on all processes
@everywhere include("../src/simulation.jl")

@everywhere begin 
    const N = length(ARGS) == 0 ? 100 : parse(Int, ARGS[1])
    const type = Float64
    const cutoff = 1e-12
end
# define global constants for parameters

const ps = collect(type, 0.0:0.05:1.0)
const ηs = collect(type, 0.0:0.05:1.0)
const param = vec([(p, η) for p in ps, η in ηs])

@everywhere begin
    const params = $param
    function calculation_multi_wrapper(lsize, idx)
        p, η = params[idx]
        return calculation_mean_multi(lsize, 4lsize, p, η; numsamp=N,
            cutoff=cutoff, ent_cutoff=cutoff, retstd=true, restype=type)
    end
end

let 
    # Model parameters
    L1, dL, L2 = 4, 2, 18
    Ls = collect(L1:dL:L2)
    nprob, neta = length(ps), length(ηs)

    h5open("data/ent_corr_data_L$(L1)_$(dL)_$(L2)_$(nprob)x$(neta).h5", "w") do file
        write(file, "datatype", string(type))
        grp = create_group(file, "params")
        write(grp, "N", N)  
        write(grp, "ps", ps)  
        write(grp, "ηs", ηs)    
        write(grp, "Ls", Ls)
    end
    
    for L in Ls
        results = pmap(idx -> calculation_multi_wrapper(L, idx), 1:nprob*neta)

        entropy_means = type[]
        entropy_stds = type[]
        corrs_means = Vector{type}[]
        corrs_stds = Vector{type}[]

        for r in results
            push!(entropy_means, r.mean_entropy)
            push!(entropy_stds, r.std_entropy)
            push!(corrs_means, r.mean_corrs)
            push!(corrs_stds, r.std_corrs)
        end

        entropy_means = reshape(entropy_means, nprob, neta)
        entropy_stds  = reshape(entropy_stds, nprob, neta)
        corrs_means = reshape(hcat(corrs_means...), nprob, neta)
        corrs_stds  = reshape(hcat(corrs_stds...), nprob, neta)

        println("L=$L done with $N samples.")
        results = nothing  # free memory

        h5open("data/ent_corr_data_L$(L1)_$(dL)_$(L2)_$(nprob)x$(neta).h5", "r+") do file
            grpL = create_group(file, "L=$L")
            grpe = create_group(grpL, "entropy_SvN")
            write(grpe, "means", entropy_means)
            write(grpe, "stds", entropy_stds)

            grpc = create_group(grpL, "correlation_Sz")
            write(grpc, "means", corrs_means)
            write(grpc, "stds", corrs_stds)
        end
        entropy_means = nothing
        entropy_stds = nothing
        corrs_means = nothing
        corrs_stds = nothing
    end
end
