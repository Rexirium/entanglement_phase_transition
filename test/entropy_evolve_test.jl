using MKL
using Statistics
using Plots, LaTeXStrings
#MKL.set_num_threads(1)

include("../src/time_evolution.jl")

MKL.set_num_threads(1)
ITensors.BLAS.set_num_threads(1)
ITensors.Strided.set_num_threads(1)

let 
    L = 24
    T = 12L
    p, η = 0.2, 0.1
    b = L ÷ 2
    
    ss = siteinds("S=1/2", L)
    psi = MPS(ss, "Up")

    obs = EntropyObserver{Float64}(b; n=1)
    @timev mps_evolve!(psi, T, p, η, obs; cutoff=1e-14, maxdim=10*L)

    entr_mean = mean(obs.entropies[2L+1:end])
    entr_std = stdm(obs.entropies[2L+1:end], entr_mean; corrected=false)
    turncerr_floor = (1e-14)*(T*L/2)

    println("Entanglement Entropy at L = $L, p=$p, η=$η : $entr_mean ± $entr_std")
    println("Truncation Error: ", obs.truncerrs[end])
    println("Truncation Error Floor: ", turncerr_floor)
    
    pt = plot(0:T, obs.truncerrs; lw = 1.5, c=:red, xaxis=L"t", yaxis="err", label=L"\epsilon_\mathrm{tot}", legend_position=:bottomright)
    plot!(0:T, fill(turncerr_floor, T+1); lw=1.5, ls=:dash, c=:black, label="trunc err floor")
    plot!(twinx(), 0:T, obs.maxbonds; lw=2, yaxis="max bond", label=L"D_\mathrm{max}", legend_position=:topright)

    pe = plot(0:T, obs.entropies, lw = 1.5, framestyle=:box, xlabel=L"t", ylabel="entropy", label=L"S_\mathrm{vN}(t)")
    
    plot(pe, pt, layout = (2,1), size=(600,800))
end