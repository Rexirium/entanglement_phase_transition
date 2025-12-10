using MKL
using ITensors, ITensorMPS
using Statistics
using Plots, LaTeXStrings

include("../src/time_evolution.jl")

let 
    L, T = 10, 200
    p, η = 0.1, 0.9
    b = L ÷ 2
    
    ss = siteinds("S=1/2", L)
    psi = MPS(ss, "Up")

    evolve = entropy_evolve!(psi, T, p, η, b)

    mean_entropy = zeros(T+1)
    for n in 1:T+1
        if n > 2L
            mean_entropy[n] = mean(evolve[2L+1:n])
        else
            continue
        end
    end


    plot(0:T, evolve, lw = 2, framestyle=:box, label=L"S_\mathrm{vN}(t)")
    plot!(0:T, mean_entropy, lw = 2, framestyle=:box, label=L"\overline{S_\mathrm{vN}}(t)")
end