using MKL
using Statistics
using Plots, LaTeXStrings
#MKL.set_num_threads(1)

include("../src/time_evolution.jl")

let 
    L, T = 16, 64
    p, η = 0.5, 0.1
    b = L ÷ 2
    
    ss = siteinds("S=1/2", L)
    psi = MPS(ss, "Up")

    evolve = @timev entropy_evolve!(psi, T, p, η, b; cutoff=eps(Float64))
    distri = [ent_entropy(psi, j, 1) for j in 0:L]

    mean_entropy = zeros(T+1)
    for n in 1:T+1
        if n > 2L
            mean_entropy[n] = mean(evolve[2L+1:n])
        else
            continue
        end
    end
    
    pe = plot(0:T, evolve, lw = 2, framestyle=:box, xlabel=L"t", label=L"S_\mathrm{vN}(t)")
    plot!(0:T, mean_entropy, lw = 2, framestyle=:box, label=L"\langle S_\mathrm{vN} \rangle(t)")
    pd = plot(0:L, distri, lw = 2, framestyle=:box, xlabel=L"x", label=L"S_\mathrm{vN}(x)")
    plot(pe, pd, layout = (2,1), size=(600,800), legend=:topright)
end