using HDF5
using CairoMakie

let 
    L = 18
    periods = 1 : 4
    ts = range(0.0, 30.0, 201)
    file = h5open("pxp_L$L", "r")

    fig = Figure(size=(800, 600))

    ax1 = Axis(fig[1, 1], xlabel=L"t", ylabel=L"S(L/2)")
    ax3 = Axis(fig[2, 1], xlabel=L"t", ylabel=L"|\langle \psi(t) | \mathbb{Z}_p \rangle|^2")

    for p in periods
        grp = file["Z_$p"]
        entropies = read(grp, "entropies")
        overlaps = read(grp, "overlaps")
        lines!(ax1, ts, entropies, label=L"| \mathbb{Z}_{%$p} \rangle")
        lines!(ax3, ts, overlaps, label=L"| \mathbb{Z}_{%$p} \rangle")
    end
    axislegend(ax1, position=:lt)
    axislegend(ax3, position=:rt)

    ax2 = Axis(fig[1, 2], xlabel=L"t", ylabel=L"C(L/2, L/2+1)")
    grp = file["Z_2"]
    correlations = read(grp, "correlations")
    lines!(ax2, ts, correlations, label=L"\mathbb{Z}_2")
    axislegend(ax2, position=:rt)

    ax4a = Axis(fig[2, 2], xlabel=L"t", ylabel=L"max bond dim", yticklabelcolor=:blue)
    ax4b = Axis(fig[2, 2], xlabel=L"t", ylabel=L"truncation error", yaxisposition=:right, yticklabelcolor=:red)
    hidespines!(ax4b)
    hidexdecorations!(ax4b)
    grp = file["Z_2"]
    maxbonds = read(grp, "maxbonds")
    truncerrs = read(grp, "truncerrs")
    lines!(ax4a, ts, maxbonds, label=L"\chi", color=:blue)
    lines!(ax4b, ts, truncerrs, label=L"\epsilon_\text{trunc}", color=:red)

    close(file)
    display(fig)

    
end