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
    ent_distr = []
    
    for p in ps
        evolve_samp = []
        distr_samp = []
        for _ in 1:num_samp
            psi, entropies = entropy_evolve(psi0, T, p, 0.5, b, 1)
            distr = [Renyi_entropy(psi, x, 1) for x in 0:L]
            push!(evolve_samp, entropies)
            push!(distr_samp, distr)
        end

        mean_evolves = sum(evolve_samp)/num_samp
        mean_distr = sum(distr_samp)/num_samp

        push!(ent_evolves, mean_entropies)
        push!(ent_distr, mean_distr)
    end
    ent_evolves = hcat(ent_evolves...)
    ent_distr = hcat(ent_distr...)

    plot(0:T, ent_evolves, 
         xlabel="Time", ylabel="Entanglement Entropy", 
         title="Entanglement Entropy Evolution for Varying p",
         label=string.(collect(ps)'), legend=:topright, framestyle=:box)
end

