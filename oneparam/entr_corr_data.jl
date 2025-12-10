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
@everywhere include("../src/simulation.jl")

let
    # Parameters
    N = length(ARGS) == 0 ? 100 : parse(Int, ARGS[1])
    type = Float64
    cutoff = 1e-12

    p0::type, η0::type = 0.5, 0.5
    ps = collect(type, 0.0:0.05:1.0)
    ηs = collect(type, 0.0:0.05:1.0)
    L1, dL, L2 = 6, 2, 18
    Ls = collect(L1:dL:L2)
    nprob, neta = length(ps), length(ηs) 

    h5open("data/entr_corr_oneparam_L$(L1)_$(dL)_$(L2)_$(nprob)x$(neta).h5", "w") do file
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
        # Calculate probability scaling in parallel using pmap
        prob_results = pmap(p ->
            calculation_mean_multi(L, 4L, p, η0; numsamp=N,
                cutoff=cutoff, retstd=true, restype=type),
            ps)

        prob_mean_entropy = [res.mean_entropy for res in prob_results]
        prob_std_entropy  = [res.std_entropy for res in prob_results]
        prob_mean_corrs = hcat([res.mean_corrs for res in prob_results]...)
        prob_std_corrs  = hcat([res.std_corrs for res in prob_results]...)

        prob_results = nothing  # free memory
        # Calculate eta scaling in parallel using pmap
        eta_results = pmap(η ->
            entropy_mean_multi(L, 4L, p0, η; numsamp=N,
                cutoff=cutoff, retstd=true, restype=type),
            ηs)

        eta_mean_entropy = [res.mean_entropy for res in eta_results]
        eta_std_entropy  = [res.std_entropy for res in eta_results]
        eta_mean_corrs = hcat([res.mean_corrs for res in eta_results]...)
        eta_std_corrs  = hcat([res.std_corrs for res in eta_results]...)

        eta_results = nothing  # free memory
        # Save data to HDF5 file
        h5open("data/entr_corr_oneparam_L$(L1)_$(dL)_$(L2)_$(nprob)x$(neta).h5", "r+") do file
            # create group if not exists
            grp = create_group(file, "results_L=$L")     
            grp1 = create_group(file, "vsprob")
            write(grp1, "mean_entropy", prob_mean_entropy)
            write(grp1, "std_entropy", prob_std_entropy)
            write(grp1, "mean_corrs", prob_mean_corrs)
            write(grp1, "std_corrs", prob_std_corrs)

            grp2 = create_group(file, "vseta")
            write(grp2, "mean_entropy", eta_mean_entropy)
            write(grp2, "std_entropy", eta_std_entropy)
            write(grp2, "mean_corrs", eta_mean_corrs)   
            write(grp2, "std_corrs", eta_std_corrs)
        end
    end
end
