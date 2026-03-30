using CairoMakie
using HDF5

let
    # Read data from HDF5 file
    file = h5open("data/time_evolve_data.h5", "r")
    ps = read(file, "params/ps")
    ηs = read(file, "params/ηs")
    T = read(file, "params/T")
    L = read(file, "params/L")
    p0 = read(file, "params/p0")
    η0 = read(file, "params/η0")
    prob_evolves = read(file, "results/prob_evolves")
    prob_distris = read(file, "results/prob_distris")
    eta_evolves = read(file, "results/eta_evolves")
    eta_distris = read(file, "results/eta_distris")
    close(file)
    

    set_theme!(Axis=(
        titlesize=20,
        xlabelsize=18,
        ylabelsize=18,
        xticklabelsize=16,
        yticklabelsize=16,
    ))

    fig = Figure(backgroundcolor = RGBf(0.98, 0.98, 0.98), size = (1000, 800))

    # Entanglement entropy time evolution for varying p
    ax1 = Axis(fig[1, 1], xlabel = L"t", ylabel = L"S(L/2)", 
        title = "entanglement entropy evolution for varying p"
    )
    for (i, p) in enumerate(ps)
        lines!(ax1, 0:T, prob_evolves[:, i], linewidth=2, label = string(p))
    end
    axislegend(ax1, position = :rt, title = L"p")

    # Entanglement entropy distribution for varying p at final time
    ax2 = Axis(fig[1, 2], xlabel = L"x", ylabel = L"S(L/2)", 
        title = "entanglement entropy distribution for varying p"
    )
    for (i, p) in enumerate(ps)
        lines!(ax2, 0:L, prob_distris[:, i], linewidth=2, label = string(p))
    end
    axislegend(ax2, position = :rt, title = L"p")

    # Entanglement entropy time evolution for varying η
    ax3 = Axis(fig[2, 1], xlabel = L"t", ylabel = L"S(L/2)", 
        title = "entanglement entropy evolution for varying η"
    )
    for (i, η) in enumerate(ηs)
        lines!(ax3, 0:T, eta_evolves[:, i], linewidth=2, label = string(η))
    end
    axislegend(ax3, position = :rt, title = L"\eta")

    # Entanglement entropy distribution for varying η at final time
    ax4 = Axis(fig[2, 2], xlabel = L"x", ylabel = L"S(L/2)", 
        title = "entanglement entropy distribution for varying η"
    )
    for (i, η) in enumerate(ηs)
        lines!(ax4, 0:L, eta_distris[:, i], linewidth=2, label = string(η))
    end
    axislegend(ax4, position = :rt, title = L"\eta")

    rowgap!(fig.layout, 5)
    colgap!(fig.layout, 5)
    
    display(fig)

    #save("figures/time_evolve_plot.png", fig)
end