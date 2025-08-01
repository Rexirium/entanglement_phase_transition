using Plots, LinearAlgebra
include("time_evolution.jl")

let 
    L = 10
    T, b = 4L, L ÷ 2
    ps = 0.0:0.2:1.0
    ηs = 0.0:0.5:2.0
    num_samp = 10

    ss = siteinds("S=1/2", L)
    psi0 = MPS(ss, "Up")

    ent_evolves = []
    for p in ps
        samples = []
        for _ in 1:num_samp
            _, entropies = entropy_evolve(psi0, T, p, 0.5, b, 1)
            push!(samples, entropies)
        end
        mean_entropies = sum(samples)/num_samp
        push!(ent_evolves, mean_entropies)
    end
    ent_evolves = hcat(ent_evolves...)
    plot(0:T, ent_evolves, 
         xlabel="Time", ylabel="Entanglement Entropy", 
         title="Entanglement Entropy Evolution for Varying p",
         label=string.(collect(ps)'), legend=:topright, framestyle=:box)
end

