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
    nsamp = 100
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

    entr_evolv_prob = Matrix{type}(undef, T+1, nprob)
    entr_distr_prob = Matrix{type}(undef, L+1, nprob)
    corr_evolv_prob = Array{type, 3}(undef, L, T+1, nprob)
    corr_distr_prob = Matrix{type}(undef, L, nprob)

    for i in 1:nprob
        entr_evolv = Matrix{type}(undef, T+1, nsamp)
        entr_distr = Matrix{type}(undef, L+1, nsamp)
        corr_evolv = Array{type, 3}(undef, L, T+1, nsamp)
        corr_distr = Matrix{type}(undef, L, nsamp)
        # Run multiple samples and average the results.
        Threads.@threads for j in 1:nsamp
            ss = siteinds("S=1/2", L)
            psi = MPS(Complex{type}, ss, "Up")
            obs = EntrCorrObserver{type}(b, L; n=1, op="Sx")
            mps_evolve!(psi, T, ps[i], η0, obs; cutoff=1e-14)

            entr_distr[:, j] .= [ent_entropy(psi, x, 1) for x in 0:L]
            corr_distr[:, j] .= correlation_vec(psi, "Sx", "Sx")

            entr_evolv[:, j] .= obs.entrs
            corr_evolv[:, :, j] .= hcat((obs.corrs)...)

            psi = nothing
            ss =nothing
            obs = nothing
        end

        entr_evolv_prob[:, i] .= sum(entr_evolv, dims=2)/nsamp
        entr_distr_prob[:, i] .= sum(entr_distr, dims=2)/nsamp
        corr_evolv_prob[:, :, i] .= sum(corr_evolv, dims=3)/nsamp
        corr_distr_prob[:, i] .= sum(corr_distr, dims=2)/nsamp
        println("Probability p = $((i-1)/(nprob-1)) done.")
    end

    h5open("data/entr_corr_evolve_L$L.h5", "r+") do file
        grp = create_group(file, "prob_results")
        write(grp, "entropy/evolve", entr_evolv_prob)
        write(grp, "entropy/distri", entr_distr_prob)
        write(grp, "correlation/evolve", corr_evolv_prob)
        write(grp, "correlation/distri", corr_distr_prob)
    end


    entr_evolv_eta = Matrix{type}(undef, T+1, neta)
    entr_distr_eta = Matrix{type}(undef, L+1, neta)
    corr_evolv_eta = Array{type, 3}(undef, L, T+1, neta)
    corr_distr_eta = Matrix{type}(undef, L, neta)

    for i in 1:neta
        entr_evolv = Matrix{type}(undef, T+1, nsamp)
        entr_distr = Matrix{type}(undef, L+1, nsamp)
        corr_evolv = Array{type, 3}(undef, L, T+1, nsamp)
        corr_distr = Matrix{type}(undef, L, nsamp)
        # Run multiple samples and average the results.
        Threads.@threads for j in 1:nsamp
            ss = siteinds("S=1/2", L)
            psi = MPS(Complex{type}, ss, "Up")
            obs = EntrCorrObserver{type}(b, L; n=1, op="Sz")
            mps_evolve!(psi, T, p0, ηs[i], obs; cutoff=1e-14)

            entr_distr[:, j] .= [ent_entropy(psi, x, 1) for x in 0:L]

            corr_distr[:, j] .= correlation_vec(psi, "Sz", "Sz")
            entr_evolv[:, j] .= obs.entrs
            corr_evolv[:, :, j] .= hcat((obs.corrs)...)

            psi = nothing
            ss =nothing
            obs = nothing
        end

        entr_evolv_eta[:, i] .= sum(entr_evolv, dims=2)/nsamp
        entr_distr_eta[:, i] .= sum(entr_distr, dims=2)/nsamp
        corr_evolv_eta[:, :, i] .= sum(corr_evolv, dims=3)/nsamp
        corr_distr_eta[:, i] .= sum(corr_distr, dims=2)/nsamp
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

