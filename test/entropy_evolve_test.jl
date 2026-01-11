using MKL
using Statistics
using Plots, LaTeXStrings
#MKL.set_num_threads(1)

include("../src/time_evolution.jl")

let 
    L, T = 16, 100
    p, η = 0.9, 0.1
    b = L ÷ 2
    
    ss = siteinds("S=1/2", L)
    psi = MPS(ss, "Up")

    evolve, truncerr, maxbonds = @timev entropy_evolve!(psi, T, p, η, b; cutoff=1e-14)
    distri = [ent_entropy(psi, j, 1) for j in 0:L]

    mean_entropy = zeros(T + 1 - 2L)
    for n in 1:T+1
        if n > 2L
            mean_entropy[n-2L] = mean(evolve[2L+1:n])
        else
            continue
        end
    end
    
    plot(0:T, truncerr; lw = 1.5, c=:red, xaxis=L"t", yaxis="err", label=L"\epsilon_\mathrm{tot}", legend_position=:bottomright)
    pt = plot!(twinx(), 0:T, maxbonds; lw=2, yaxis="max bond", label=L"D_\mathrm{max}", legend_position=:topright)

    pe = plot(0:T, evolve, lw = 1.5, framestyle=:box, xlabel=L"t", ylabel="entropy", label=L"S_\mathrm{vN}(t)")
    plot!((2L):T, mean_entropy, lw = 2, framestyle=:box, label=L"\langle S_\mathrm{vN} \rangle(t)")
    #pd = plot(0:L, distri, lw = 2, framestyle=:box, xlabel=L"x", label=L"S_\mathrm{vN}(x)")
    plot(pe, pt, layout = (2,1), size=(600,800))
end