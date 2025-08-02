using Plots, LinearAlgebra
include("time_evolution.jl")

let 
    L = 10
    T, b = 4L, L ÷ 2
    ps = 0.0:0.2:1.0
    ηs = 0.0:0.5:2.0
    numsamp = 10

    ss = siteinds("S=1/2", L)
    psi0 = MPS(ss, "Up")

    ent_evolves = []
    ent_distris = []

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

        push!(ent_evolves, meanevolve)
        push!(ent_distris, meandistri)
    end
    ent_evolves = hcat(ent_evolves...)
    ent_distris = hcat(ent_distris...)

    plot(0:T, ent_evolves, 
         xlabel="Time", ylabel="Entanglement Entropy", 
         title="Entanglement Entropy Evolution for Varying p",
         label=string.(collect(ps)'), legend=:topright, framestyle=:box)
end

