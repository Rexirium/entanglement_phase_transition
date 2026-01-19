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
    const cutoff = 1e-14
end
# define global constants for parameters

const ps = collect(type, 0.0:0.05:1.0)
const ηs = collect(type, 0.05:0.05:1.0)
const param = vec([(p, η) for p in ps, η in ηs])

@everywhere begin
    const params = $param
    function entrcorr_average_wrapper(lsize::Int, ttotal::Int, idx::Int)
        p, η = params[idx]
        ss = siteinds("S=1/2", lsize)
        psi = MPS(Complex{type}, ss, "Up")
        avg = EntrCorrAverager{type}(lsize ÷ 2, lsize; n=1, op="Sz")
        # core calculation
        mps_evolve!(psi, ttotal, p, η, avg; cutoff=cutoff)
        return avg
    end
end

let 
    # Model parameters
    L1, dL, L2 = 4, 2, 20
    Ls = collect(L1:dL:L2)
    nprob, neta = length(ps), length(ηs)

    h5open("data/entrcorr_avg_L$(L1)_$(dL)_$(L2)_$(nprob)x$(neta).h5", "w") do file
        write(file, "datatype", string(type))
        grp = create_group(file, "params")
        write(grp, "N", N)
        write(grp, "ps", ps)
        write(grp, "ηs", ηs)
        write(grp, "Ls", Ls)
    end

    for L in Ls
        T = 12L
        nparams = length(params)
        averagers = pmap(idx -> entrcorr_average_wrapper(L, T, idx), 1:nparams)

        entr_means = type[]
        entr_stds = type[]
        corr_means = Vector{type}[]
        corr_stds = Vector{type}[]

        for avg in averagers
            push!(entr_means, avg.entr_mean)
            push!(entr_stds, sqrt(avg.entr_sstd / (T - 2L)))
            push!(corr_means, avg.corr_mean)
            push!(corr_stds, sqrt.(avg.corr_std ./ (T - 2L)))
        end

        entr_means = reshape(entr_means, nprob, neta)
        entr_stds  = reshape(entr_stds, nprob, neta)
        corr_means = reshape(hcat(corr_means...), L, nprob, neta)
        corr_stds  = reshape(hcat(corr_stds...), L, nprob, neta)

        println("L=$L done with $N samples.")
        averagers = nothing  # free memory

        h5open("data/entrcorr_avg_L$(L1)_$(dL)_$(L2)_$(nprob)x$(neta).h5", "r+") do file
            grpL = create_group(file, "L_$L")
            write(grpL, "entr_means", entr_means)
            write(grpL, "entr_stds", entr_stds)
            write(grpL, "corr_means", corr_means)
            write(grpL, "corr_stds", corr_stds)
        end
    end   
end