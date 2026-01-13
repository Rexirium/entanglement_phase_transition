using MKL
using Statistics
using Plots, LaTeXStrings
#MKL.set_num_threads(1)

include("../src/time_evolution.jl")

let 
    L, T = 16, 100
    p, η = 0.5, 0.5
    b = L ÷ 2
    
    ss = siteinds("S=1/2", L)
    psi = MPS(ss, "Up")

    obs = EntropyObserver{Float64}(b; n=1)
    truncerr = mps_evolve!(psi, T, p, η, obs; cutoff=eps(Float64))
    
    plot(0:T, obs.truncerrs; lw = 1.5, c=:red, xaxis=L"t", yaxis="err", label=L"\epsilon_\mathrm{tot}", legend_position=:bottomright)
    pt = plot!(twinx(), 0:T, obs.maxbonds; lw=2, yaxis="max bond", label=L"D_\mathrm{max}", legend_position=:topright)

    pe = plot(0:T, obs.entropies, lw = 1.5, framestyle=:box, xlabel=L"t", ylabel="entropy", label=L"S_\mathrm{vN}(t)")
    
    plot(pe, pt, layout = (2,1), size=(600,800))
end