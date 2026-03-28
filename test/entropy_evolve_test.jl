using MKL
using ITensors, ITensorMPS
ITensors.BLAS.set_num_threads(1)
ITensors.Strided.set_num_threads(1)
using Statistics
using CairoMakie

if !isdefined(Main, :RandomUnitary)
    include("../src/RandomUnitary.jl")
    using .RandomUnitary
end

let 
    L = 16
    T = 12L
    cutoff = eps(Float64)
    p, η = 0.5, 0.1
    b = L ÷ 2
    
    dent = NHDisentangler{Float64}(p, η)
    ss = siteinds("S=1/2", L)
    psi = MPS(ComplexF64, ss, "Up")

    obs = EntropyObserver{Float64}(b; n=1)
    Dm = 25*L
    threshold = 1e-8 * (T*L)
    @timev mps_evolve!(psi, T, dent, obs; cutoff=cutoff, maxdim=Dm)
    tsteps = length(obs.entropies) - 1

    if tsteps < T
        println("Evolution stopped early at t = $tsteps due to high truncation error.")
    else
        entr_mean = mean(obs.entropies[2L+2:end])
        entr_sem = stdm(obs.entropies[2L+2:end], entr_mean) / (T - 2L)

        println("Entanglement Entropy at L = $L, p=$p, η=$η : $entr_mean ± $entr_sem")
        println("Truncation Error: ", obs.truncerrs[end])
        println("Truncation Error Threshold: ", threshold)
    end

    # 1. Initialize Figure with specified size
    fig = Figure(size = (600, 800))

    # 2. Top Plot: Entropies
    ax_entropy = Axis(fig[1, 1],
        xlabel = L"t",
        ylabel = L"S",
        title = L"L = %$L, p= %$p, \eta= %$η"
    )

    lines!(ax_entropy, 0:T, obs.entropies, 
        linewidth = 1.5, 
        label = L"S_\mathrm{vN}(t)"
    )
    axislegend(ax_entropy, position = :rt) # rt = right-top

    # 3. Bottom Plot: Max Bonds
    ax_bond = Axis(fig[2, 1],
        ylabel = L"D_\mathrm{max}",
        xlabel = L"t", # Standard practice to add xlabel to the bottom plot
    )

    lines!(ax_bond, 0:tsteps, obs.maxbonds, 
        linewidth = 2, 
        label = "max bond"
    )
    axislegend(ax_bond, position = :rt)

    # Optional: Tighten the layout to handle margins automatically
    colgap!(fig.layout, 10)
    rowgap!(fig.layout, 10)

    display(fig)
end