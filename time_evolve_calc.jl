using HDF5
include("time_evolution.jl")

let 
    L = 10
    T, b = 4L, L ÷ 2
    ps = 0.0:0.2:1.0
    ηs = 0.0:0.5:2.0
    numsamp = 10

    ss = siteinds("S=1/2", L)
    psi0 = MPS(ss, "Up")

    prob_evolves = []
    prob_distris = []

    for p in ps
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

        push!(prob_evolves, meanevolve)
        push!(prob_distris, meandistri)
    end
    prob_evolves = hcat(prob_evolves...)
    prob_distris = hcat(prob_distris...)

    eta_evolves = []
    eta_distris = []

    for η in ηs
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

        push!(eta_evolves, meanevolve)
        push!(eta_distris, meandistri)
    end
    eta_evolves = hcat(eta_evolves...)
    eta_distris = hcat(eta_distris...) 

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

