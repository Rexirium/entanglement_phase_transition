using MKL
using ITensors, ITensorMPS
using Statistics
using Plots, LaTeXStrings

include("../src/time_evolution.jl")

let 
    L, T = 10, 200
    p, η = 0.5, 0.5
    b = L ÷ 2
    
    ss = siteinds("S=1/2", L)
    psi = MPS(ss, "Up")

    evolve = entropy_evolve!(psi, T, p, η, b)

    mean_entropy = zeros(T+1)
    for n in 1:T+1
        mean_entropy[n] = mean(evolve[1:n])
    end


    plot(0:T, evolve, lw = 2, framestyle=:box, label=L"S_\mathrm{vN}(t)")
    plot!(0:T, mean_entropy, lw = 2, framestyle=:box, label=L"\overline{S_\mathrm{vN}}(t)")
end