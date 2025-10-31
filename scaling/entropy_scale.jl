using MKL
using HDF5
using Distributed

MKL.set_num_threads(1)

# add worker processes if none exist (use CPU-1 workers to avoid oversubscription)
if nprocs() == 1
    nworkers_to_add = max(Sys.CPU_THREADS ÷ Threads.nthreads() - 4, 1)
    addprocs(nworkers_to_add)
end

# make sure workers have required packages and the same MKL threading setting
@everywhere using MKL
@everywhere MKL.set_num_threads(1)

# include the entropy calculation code on all processes
@everywhere include("../src/entropy_calc.jl")

let
    # Parameters
    N = length(ARGS) == 0 ? 100 : parse(Int, ARGS[1])
    type = Float64

    p0::type, η0::type = 0.5, 0.5
    ps = collect(type, 0.0:0.05:1.0)
    ηs = collect(type, 0.0:0.05:1.0)
    L1, dL, L2 = 6, 2, 18
    Ls = collect(L1:dL:L2)
    nprob, neta = length(ps), length(ηs)

    h5open("data/entropy_scale_L$(L1)_$(dL)_$(L2).h5", "w") do file
        write(file, "datatype", string(type))
        grp = create_group(file, "params")
        write(grp, "N", N)  
        write(grp, "p0", p0)  
        write(grp, "η0", η0) 
        write(grp, "ps", ps)  
        write(grp, "ηs", ηs)    
        write(grp, "Ls", Ls)
    end

    for L in Ls
        cutoff = 1e-12 * L^3

        # Calculate probability scaling in parallel using pmap
        prob_results = pmap(p ->
            entropy_mean_multi(L, 4L, p, η0; numsamp=N,
                cutoff=cutoff, ent_cutoff=cutoff, retstd=true, restype=type),
            ps)

        prob_scales_mean = [r[1] for r in prob_results]
        prob_scales_std  = [r[2] for r in prob_results]

        for i in 1:nprob
            println("L=$L, p=$(round(ps[i],digits=2)), η=0.5 done with $N samples.")
        end
        prob_results = nothing  # free memory
        # Calculate eta scaling in parallel using pmap
        eta_results = pmap(η ->
            entropy_mean_multi(L, 4L, p0, η; numsamp=N,
                cutoff=cutoff, ent_cutoff=cutoff, retstd=true, restype=type),
            ηs)

        eta_scales_mean = [r[1] for r in eta_results]
        eta_scales_std  = [r[2] for r in eta_results]

        for i in 1:neta
            println("L=$L, p=0.50, η=$(round(ηs[i],digits=2)) done with $N samples.")
        end
        eta_results = nothing  # free memory
        # Save data to HDF5 file
        h5open("data/entropy_scale_L$(L1)_$(dL)_$(L2).h5", "cw") do file
            # create group if not exists
            grp = create_group(file, "results_L=$L")     

            write(grp, "prob_scales_mean", prob_scales_mean)
            write(grp, "prob_scales_std", prob_scales_std)
            write(grp, "eta_scales_mean", eta_scales_mean)
            write(grp, "eta_scales_std", eta_scales_std)
        end
    end
end
