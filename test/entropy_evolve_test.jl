using MKL
using Statistics
using Plots, LaTeXStrings
#MKL.set_num_threads(1)

include("../src/time_evolution.jl")

MKL.set_num_threads(1)
ITensors.BLAS.set_num_threads(1)
ITensors.Strided.set_num_threads(1)

let 
    L = 20
    T = 12L
    cutoff = eps(Float64)
    p, η = 0.5, 0.1
    b = L ÷ 2
    
    dent = NHDisentangler{Float64}(p, η)
    ss = siteinds("S=1/2", L)
    psi = MPS(ComplexF64, ss, "Up")

    obs = EntropyObserver{Float64}(b; n=1)
    Dm = 100 + 10*L
    threshold = 1e-8 * (T*L)
    @timev mps_evolve!(psi, T, dent, obs; cutoff=cutoff, maxdim=Dm, etol=threshold)
    tsteps = length(obs.entropies) - 1

    if tsteps < T
        println("Evolution stopped early at t = $tsteps due to high truncation error.")
    else
        entr_mean = mean(obs.entropies[2L+2:end])
        entr_std = stdm(obs.entropies[2L+2:end], entr_mean; corrected=false)

        println("Entanglement Entropy at L = $L, p=$p, η=$η : $entr_mean ± $entr_std")
        println("Truncation Error: ", obs.truncerrs[end])
        println("Truncation Error Threshold: ", threshold)
    end

    pbond = plot(0:tsteps, obs.maxbonds; lw =2, yaxis=L"D_\mathrm{max}", 
        label="max bond", framestyle=:box, title=latexstring("L = $L, p=$p, η=$η"))
    hline!([Dm], lw=2, label="max bond limit")

    perr = plot(0:tsteps, obs.truncerrs; lw = 1.5, xaxis=L"t", label="truncation error", framestyle=:box)
    hline!([threshold], lw=1.5, l=:dash, label="trunc err ceiling")

    #pe = plot(0:T, obs.entropies, lw = 1.5, framestyle=:box, xlabel=L"t", ylabel="entropy", label=L"S_\mathrm{vN}(t)")
    
    plot(pbond, perr, layout = (2,1), size=(600,800), left_margin=4Plots.mm)
end