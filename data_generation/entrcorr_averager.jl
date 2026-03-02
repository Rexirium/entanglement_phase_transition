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

const nprob = 21
const neta = 21
const ps = LinRange{type}(0.0, 1.0, nprob)
const ηs = LinRange{type}(0.0, 1.0, neta)
const param = vec([(p, η) for p in ps, η in ηs])

@everywhere begin
    const params = $param

    function entrcorr_average_wrapper(lsize::Int, ttotal::Int, idx::Int)
        dent = NHDisentangler{type}(params[idx]...)
        ss = siteinds("S=1/2", lsize)
        psi = MPS(Complex{type}, ss, "Up")
        avg = EntrCorrAverager{type}(lsize ÷ 2, lsize; n=1, op="Sz")
        # core calculation
        threshold = 1e-6 * (ttotal * lsize)
        maxbond = 20*lsize
        truncerr = mps_evolve!(psi, ttotal, dent, avg; cutoff=cutoff, maxdim=maxbond, etol=threshold)

        psi = nothing  # free memory
        return avg, truncerr
    end
end

let 
    # Model parameters
    L1, dL, L2 = 10, 2, 40
    Ls = L1:dL:L2
    nparam = length(params)
    subs = CartesianIndices((nprob, neta))

    h5open("data/nh_entrcorr_avg_L$(L1)_$(dL)_$(L2)_$(nprob)x$(neta).h5", "w") do file
        write(file, "datatype", string(type))
        grp = create_group(file, "params")
        write(grp, "ps", collect(ps))
        write(grp, "ηs", collect(ηs))
        write(grp, "Ls", collect(Ls))
    end

    for L in Ls
        T = 12L
        N = T - 2L

        averagers = pmap(idx -> entrcorr_average_wrapper(L, T, idx), 1:nparam)

        entr_means  = Matrix{type}(undef, nprob, neta)
        entr_sems   = Matrix{type}(undef, nprob, neta)
        corr_means  = Array{type, 3}(undef, L, nprob, neta)
        corr_sems   = Array{type, 3}(undef, L, nprob, neta)
        truncerrs   = Matrix{type}(undef, nprob, neta)

        for idx in 1:nparam
            avg, truncerr = averagers[idx]
            sub = subs[idx]

            if avg.accept
                entr_means[sub] = avg.entr_mean
                entr_sems[sub] = sqrt(avg.entr_sstd / (N*(N-1)))
                corr_means[:, sub] = avg.corr_mean
                corr_sems[:, sub] = sqrt.(avg.corr_sstd / (N*(N-1)))
                truncerrs[sub] = truncerr
            else
                entr_means[sub] = NaN
                entr_sems[sub] = NaN
                corr_means[:, sub] .= NaN
                corr_sems[:, sub] .= NaN
                truncerrs[sub] = NaN
            end
        end

        println("L=$L done with $N samples.")
        averagers = nothing  # free memory

        h5open("data/nh_entrcorr_avg_L$(L1)_$(dL)_$(L2)_$(nprob)x$(neta).h5", "r+") do file
            grpL = create_group(file, "L=$L")
            write(grpL, "entr_means", entr_means)
            write(grpL, "entr_sems", entr_sems)
            write(grpL, "corr_means", corr_means)
            write(grpL, "corr_sems", corr_sems)
            write(grpL, "truncerrs", truncerrs)
        end
    end   
end