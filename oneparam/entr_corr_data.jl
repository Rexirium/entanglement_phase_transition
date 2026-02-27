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
@everywhere begin
    using MKL
    MKL.set_num_threads(1)
    using Statistics
    # include the entropy calculation code on all processes
    include("../src/simulation.jl")
    ITensors.BLAS.set_num_threads(1)
    ITensors.Strided.set_num_threads(1)

    function entrcorr_mean_multi(lsize::Int, ttotal::Int, p::Real, η::Real; 
        nsamp::Int=100, cutoff::Real=1e-14, restype=Float64)
        res = EntrCorrResults{restype}(lsize ÷ 2, lsize; n=1, op="Sx", nsamp=nsamp)
        calculation_mean_multi(lsize, ttotal, p, η, res; cutoff=cutoff)

        entr_mean = mean(res.entropies)
        entr_std = stdm(res.entropies, entr_mean) / sqrt(nsamp)
        corr_mean = vec(mean(res.corrs, dims=2))
        corr_std = vec(stdm(res.corrs, corr_mean; dims=2)) / sqrt(nsamp)
        return CalcResult{restype}(entr_mean, entr_std, corr_mean, corr_std)
    end
end

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
            entrcorr_mean_multi(L, 4L, p, η0; nsamp=N,
                cutoff=cutoff, restype=type),
            ps)

        prob_mean_entropy = [res.entr_mean for res in prob_results]
        prob_std_entropy  = [res.entr_std for res in prob_results]
        prob_mean_corrs = hcat([res.corr_mean for res in prob_results]...)
        prob_std_corrs  = hcat([res.corr_std for res in prob_results]...)

        prob_results = nothing  # free memory
        # Calculate eta scaling in parallel using pmap
        eta_results = pmap(η ->
            entrcorr_mean_multi(L, 4L, p0, η; nsamp=N,
                cutoff=cutoff, restype=type),
            ηs)

        eta_mean_entropy = [res.entr_mean for res in eta_results]
        eta_std_entropy  = [res.entr_std for res in eta_results]
        eta_mean_corrs = hcat([res.corr_mean for res in eta_results]...)
        eta_std_corrs  = hcat([res.corr_std for res in eta_results]...)

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
