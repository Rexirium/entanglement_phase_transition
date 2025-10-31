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
    ηs = collect(type, 0.0:0.5:2.0)
    p0::type, η0::type = 0.5, 0.5
    numsamp = 100
    nprob, neta = length(ps), length(ηs)

    h5open("data/time_evolve_data.h5", "w") do file
        write(file, "datatype", string(type))
        grp = create_group(file, "params")
        write(grp, "L", L)
        write(grp, "T", T)
        write(grp, "p0", p0)
        write(grp, "η0", η0)
        write(grp, "ps", ps)
        write(grp, "ηs", ηs)
    end

    prob_evolves = zeros(type, T+1, nprob)
    prob_distris = zeros(type, L+1, nprob)

    for i in 1:nprob
        evolvesamp = Matrix{type}(undef, T+1, numsamp)
        distrisamp = Matrix{type}(undef, L+1, numsamp)
        # Run multiple samples and average the results.
        Threads.@threads for j in 1:numsamp
            ss = siteinds("S=1/2", L)
            psi = MPS(Complex{type}, ss, "Up")
            evolve = entropy_evolve!(psi, T, ps[i], η0, b, 1)
            distri = [Renyi_entropy(psi, x, 1) for x in 0:L]
            evolvesamp[:, j] .= evolve
            distrisamp[:, j] .= distri
            psi = nothing
            ss =nothing
            evlvoe = nothing
            distri = nothing
        end

        meanevolve = sum(evolvesamp, dims=2)/numsamp
        meandistri = sum(distrisamp, dims=2)/numsamp

        prob_evolves[:, i] .= meanevolve
        prob_distris[:, i] .= meandistri
    end


    eta_evolves = zeros(type, T+1, neta)
    eta_distris = zeros(type, L+1, neta)

    for i in 1:neta
        evolvesamp = Matrix{type}(undef, T+1, numsamp)
        distrisamp = Matrix{type}(undef, L+1, numsamp)
        # Run multiple samples and average the results.
        Threads.@threads for j in 1:numsamp
            ss = siteinds("S=1/2", L)
            psi = MPS(Complex{type}, ss, "Up")
            evolve = entropy_evolve!(psi, T, p0, ηs[i], b, 1)
            distri = [Renyi_entropy(psi, x, 1) for x in 0:L]
            evolvesamp[:, j] .= evolve
            distrisamp[:, j] .= distri
            psi = nothing
            ss =nothing
            evlvoe = nothing
            distri = nothing
        end

        meanevolve = sum(evolvesamp, dims=2)/numsamp
        meandistri = sum(distrisamp, dims=2)/numsamp

        eta_evolves[:, i] .= meanevolve
        eta_distris[:, i] .= meandistri
    end

    h5open("data/time_evolve_data.h5", "r+") do file
        grp = create_group(file, "results")
        write(grp, "prob_evolves", prob_evolves)
        write(grp, "prob_distris", prob_distris)
        write(grp, "eta_evolves", eta_evolves)
        write(grp, "eta_distris", eta_distris)
    end
end

