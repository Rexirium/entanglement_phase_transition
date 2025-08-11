using HDF5
using Base.Threads
include("time_evolution.jl")
nthreads() = 4
BLAS.set_num_threads(nthreads())

let 
    L = 10
    T, b = 4L, L ÷ 2
    ps = 0.0:0.2:1.0
    ηs = 0.0:0.5:2.0

    numsamp = 10
    nprob = length(ps)
    neta = length(ηs)

    prob_evolves = zeros(T+1, nprob)
    prob_distris = zeros(L+1, nprob)

    @threads for i = 1:nprob
        p = ps[i]
        ss = siteinds("S=1/2", L)
        psi0 = MPS(ss, "Up")

        evolvesamp = []
        distrisamp = []
        for _ in 1:numsamp
            psi, evolve = entropy_evolve(psi0, T, p, 0.5, b, 1)
            distri = [Renyi_entropy(psi, x, 1) for x in 0:L]
            push!(evolvesamp, evolve)
            push!(distrisamp, distri)
        end

        meanevolve = sum(evolvesamp)/numsamp
        meandistri = sum(distrisamp)/numsamp

        prob_evolves[:, i] = meanevolve
        prob_distris[:, i] = meandistri
    end

    eta_evolves = zeros(T+1, neta)
    eta_distris = zeros(L+1, neta)

    @threads for i = 1:neta
        η = ηs[i]
        ss = siteinds("S=1/2", L)
        psi0 = MPS(ss, "Up")

        evolvesamp = []
        distrisamp = []
        for _ in 1:numsamp
            psi, evolve = entropy_evolve(psi0, T, 0.5, η, b, 1)
            distri = [Renyi_entropy(psi, x, 1) for x in 0:L]
            push!(evolvesamp, evolve)
            push!(distrisamp, distri)
        end

        meanevolve = sum(evolvesamp)/numsamp
        meandistri = sum(distrisamp)/numsamp

        eta_evolves[:, i] = meanevolve
        eta_distris[:, i] = meandistri
    end

    h5open("time_evolve_data.h5", "w") do file
        write(file, "ps", collect(ps))
        write(file, "ηs", collect(ηs))
        write(file, "L", L)
        write(file, "T", T)
        write(file, "prob_evolves", prob_evolves)
        write(file, "prob_distris", prob_distris)
        write(file, "eta_evolves", eta_evolves)
        write(file, "eta_distris", eta_distris)
    end
end

