using MKL
using HDF5
using ITensors, ITensorMPS
MKL.set_num_threads(1)
ITensors.BLAS.set_num_threads(1)
ITensors.Strided.set_num_threads(1)

include("../src/time_evolution.jl")

let 
    # Parameters
    type = Float64
    L = 10
    T, b = 4L, L ÷ 2
    ps = collect(type, 0.0:0.2:1.0)
    ηs = collect(type, 0.0:0.2:1.0)
    p0::type, η0::type = 0.5, 0.5
    numsamp = 100
    nprob, neta = length(ps), length(ηs)

    h5open("data/entr_corr_evolve_L$L.h5", "w") do file
        write(file, "datatype", string(type))
        grp = create_group(file, "params")
        write(grp, "L", L)
        write(grp, "T", T)
        write(grp, "p0", p0)
        write(grp, "η0", η0)
        write(grp, "ps", ps)
        write(grp, "ηs", ηs)
    end

    entr_evolv_prob = zeros(type, T+1, nprob)
    entr_distr_prob = zeros(type, L+1, nprob)
    corr_evolv_prob = zeros(type, L, T+1, nprob)
    corr_distr_prob = zeros(type, L, nprob)

    for i in 1:nprob
        entr_evolv = Matrix{type}(undef, T+1, numsamp)
        entr_distr = Matrix{type}(undef, L, L+1, numsamp)
        corr_evolv = Array{type, 3}(undef, L, T+1, numsamp)
        corr_distr = Matrix{type}(undef, L, numsamp)
        # Run multiple samples and average the results.
        Threads.@threads for j in 1:numsamp
            ss = siteinds("S=1/2", L)
            psi = MPS(Complex{type}, ss, "Up")
            entr, corr = entr_corr_evolve!(psi, T, ps[i], η0, b)
            entr_distr = [Renyi_entropy(psi, x, 1) for x in 0:L]

            corr_distr[:, j] .= correlation_vec(psi, "Sz", "Sz")
            entr_evolv[:, j] .= entr
            corr_evolv[:, :, j] .= corr
            entr_distr[:, j] .= entr_distr

            psi = nothing
            ss =nothing
        end

        entr_evolv_prob[:, i] .= sum(entr_evolv, dims=2)/numsamp
        entr_distr_prob[:, i] .= sum(entr_distr, dims=2)/numsamp
        corr_evolv_prob[:, :, i] .= sum(corr_evolv, dims=3)/numsamp
        corr_distr_prob[:, i] .= sum(corr_distr, dims=2)/numsamp
        println("Probability p = $((i-1)/(nprob-1)) done.")
    end

    h5open("data/entr_corr_evolve_L$L.h5", "r+") do file
        grp = create_group(file, "prob_results")
        write(grp, "entropy/evolve", entr_evolv_prob)
        write(grp, "entropy/distri", entr_distr_prob)
        write(grp, "correlation/evolve", corr_evolv_prob)
        write(grp, "correlation/distri", corr_distr_prob)
    end


    entr_evolv_eta = zeros(type, T+1, neta)
    entr_distr_eta = zeros(type, L+1, neta)
    corr_evolv_eta = zeros(type, L, T+1, neta)
    corr_distr_eta = zeros(type, L, neta)

    for i in 1:neta
        entr_evolv = Matrix{type}(undef, T+1, numsamp)
        entr_distr = Matrix{type}(undef, L, L+1, numsamp)
        corr_evolv = Array{type, 3}(undef, L, T+1, numsamp)
        corr_distr = Matrix{type}(undef, L, numsamp)
        # Run multiple samples and average the results.
        Threads.@threads for j in 1:numsamp
            ss = siteinds("S=1/2", L)
            psi = MPS(Complex{type}, ss, "Up")
            entr, corr = entr_corr_evolve!(psi, T, p0, ηs[i], b)
            entr_distr = [Renyi_entropy(psi, x, 1) for x in 0:L]

            corr_distr[:, j] .= correlation_vec(psi, "Sz", "Sz")
            entr_evolv[:, j] .= entr
            corr_evolv[:, :, j] .= corr
            entr_distr[:, j] .= entr_distr

            psi = nothing
            ss =nothing
        end

        entr_evolv_eta[:, i] .= sum(entr_evolv, dims=2)/numsamp
        entr_distr_eta[:, i] .= sum(entr_distr, dims=2)/numsamp
        corr_evolv_eta[:, :, i] .= sum(corr_evolv, dims=3)/numsamp
        corr_distr_eta[:, i] .= sum(corr_distr, dims=2)/numsamp
        println("Non Hermitian η = $((i-1)/(neta-1)) done.")
    end

    h5open("data/entr_corr_evolve_L$L.h5", "r+") do file
        grp = create_group(file, "eta_results")
        write(grp, "entropy/evolve", entr_evolv_eta)
        write(grp, "entropy/distri", entr_distr_eta)
        write(grp, "correlation/evolve", corr_evolv_eta)
        write(grp, "correlation/distri", corr_distr_eta)
    end
end

