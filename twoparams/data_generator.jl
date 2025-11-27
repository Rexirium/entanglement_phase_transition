using MKL
using HDF5
using Distributed

MKL.set_num_threads(1)

# add worker processes if none exist (use CPU-1 workers to avoid oversubscription)
if nprocs() == 1
    nworkers_to_add = max(Sys.CPU_THREADS ÷ Threads.nthreads() - 2, 1)
    addprocs(nworkers_to_add)
end
# make sure workers have required packages and the same MKL threading setting
@everywhere using MKL
@everywhere MKL.set_num_threads(1)

# include the entropy calculation code on all processes
@everywhere include("../src/entropy_calc.jl")

@everywhere const N = length(ARGS) == 0 ? 100 : parse(Int, ARGS[1])
# define global constants for parameters
const type = Float64
const ps = collect(type, 0.0:0.05:1.0)
const ηs = collect(type, 0.0:0.05:1.0)
const param = vec([(p, η) for p in ps, η in ηs])

@everywhere begin
    const params = $param

    function entropy_mean_multi_wrapper(lsize, idx)
        p, η = params[idx]
        cutoff = 1e-12
        return entropy_mean_multi(lsize, 4lsize, p, η; numsamp=N,
            cutoff=cutoff, ent_cutoff=cutoff, retstd=true, restype=type)
    end
end

let 
    # Model parameters
    L1, dL, L2 = 4, 2, 18
    Ls = collect(L1:dL:L2)
    nprob, neta = length(ps), length(ηs)

    h5open("data/entropy_data_L$(L1)_$(dL)_$(L2)_$(nprob)x$(neta).h5", "w") do file
        write(file, "datatype", string(type))
        grp = create_group(file, "params")
        write(grp, "N", N)  
        write(grp, "ps", ps)  
        write(grp, "ηs", ηs)    
        write(grp, "Ls", Ls)
    end
    
    for L in Ls
        results = pmap(idx -> entropy_mean_multi_wrapper(L, idx), 1:nprob*neta)

        data_means = reshape([r[1] for r in results], nprob, neta)
        data_stds  = reshape([r[2] for r in results], nprob, neta)

        println("L=$L done with $N samples.")
        results = nothing  # free memory

        h5open("data/entropy_data_L$(L1)_$(dL)_$(L2)_$(nprob)x$(neta).h5", "r+") do file
            grpL = create_group(file, "L=$L")
            write(grpL, "means", data_means)
            write(grpL, "stds", data_stds)
        end
        data_means = nothing
        data_stds = nothing
    end
end
