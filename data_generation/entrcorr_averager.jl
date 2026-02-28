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
end
# define global constants for parameters

const ps = collect(type, 0.0:0.05:1.0)
const ηs = prepend(collect(type, 0.05:0.05:1.0), type(0.01))
const param = vec([(p, η) for p in ps, η in ηs])

@everywhere begin
    const params = $param

    function entrcorr_average_wrapper(lsize::Int, ttotal::Int, idx::Int)
        dent = NHDisentangler{type}(params[idx]...)
        ss = siteinds("S=1/2", lsize)
        psi = MPS(Complex{type}, ss, "Up")
        avg = EntrCorrAverager{type}(lsize ÷ 2, lsize; n=1, op="Sz")
        # core calculation
        threshold = 1e-8 * (ttotal * lsize)
        maxbond = 20*lsize
        truncerr = mps_evolve!(psi, ttotal, dent, avg; cutoff=cutoff, maxdim=maxbond, etol=threshold)

        psi = nothing  # free memory
        return avg, truncerr
    end
end

let 
    # Model parameters
    L1, dL, L2 = 8, 4, 40
    Ls = collect(L1:dL:L2)
    nprob, neta = length(ps), length(ηs)

    h5open("data/nh_entrcorr_avg_L$(L1)_$(dL)_$(L2)_$(nprob)x$(neta).h5", "w") do file
        write(file, "datatype", string(type))
        grp = create_group(file, "params")
        write(grp, "ps", ps)
        write(grp, "ηs", ηs)
        write(grp, "Ls", Ls)
    end

    for L in Ls
        T = 10L
        N = T - 2L
        nparams = length(params)
        averagers = pmap(idx -> entrcorr_average_wrapper(L, T, idx), 1:nparams)

        entr_means = type[]
        entr_sems = type[]
        corr_means = Vector{type}[]
        corr_sems = Vector{type}[]
        truncerrs = type[]

        for (avg, truncerr) in averagers
            if avg.accept
                push!(entr_means, avg.entr_mean)
                push!(entr_sems, sqrt(avg.entr_sstd / (N*(N-1))))
                push!(corr_means, avg.corr_mean)
                push!(corr_sems, sqrt.(avg.corr_sstd ./ (N*(N-1))))
                push!(truncerrs, truncerr)
            else
                push!(entr_means, NaN)
                push!(entr_sems, NaN)
                push!(corr_means, fill(NaN, L))
                push!(corr_sems, fill(NaN, L))
                push!(truncerrs, NaN)
            end
        end

        entr_means = reshape(entr_means, nprob, neta)
        entr_sems  = reshape(entr_sems, nprob, neta)
        corr_means = reshape(hcat(corr_means...), L, nprob, neta)
        corr_sems  = reshape(hcat(corr_sems...), L, nprob, neta)
        truncerrs = reshape(truncerrs, nprob, neta)

        println("L=$L done with $(8L) samples.")
        averagers = nothing  # free memory

        h5open("data/nh_entrcorr_avg_L$(L1)_$(dL)_$(L2)_$(nprob)x$(neta).h5", "r+") do file
            grpL = create_group(file, "L_$L")
            write(grpL, "entr_means", entr_means)
            write(grpL, "entr_sems", entr_sems)
            write(grpL, "corr_means", corr_means)
            write(grpL, "corr_sems", corr_sems)
            write(grpL, "truncerrs", truncerrs)
        end
    end   
end