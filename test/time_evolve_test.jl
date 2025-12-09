using MKL
using ITensors, ITensorMPS
using Plots

include("../src/time_evolution.jl")

let 
    L, T = 10, 80
    p, η = 0.5, 0.5
    b = L ÷ 2
    
    ss = siteinds("S=1/2", L)
    psi = MPS(ss, "Up")

    evolve = entropy_evolve!(psi, T, p, η, b)

    plot(0:T, evolve, lw = 2, framestyle=:box)
end