using HDF5
using CairoMakie

let 
    L = 18
    
    file = h5open("manybody_scars/pxp_L$(L).h5", "r")

    nsteps = read(file, "params/nsteps")
    ps = read(file, "params/periods")
    ts = range(0.0, 30.0, nsteps)

    set_theme!(Axis=(
        xgridvisible=false, ygridvisible=false,
        xtickalign=1, ytickalign=1,
        xlabelsize=18, ylabelsize=18,
        xticklabelsize=16, yticklabelsize=16
    ), theme_latexfonts())

    mycolors = copy(Makie.wong_colors())
    mycolors[1], mycolors[2] = mycolors[2], mycolors[1]

    fig = Figure(size=(600, 600))

    ax1 = Axis(fig[1, 1], ylabel=L"|\langle \psi(t) | \mathbb{Z}_p \rangle|^2", 
        palette=(color=mycolors,))
    ax2 = Axis(fig[2, 1], xlabel=L"t", ylabel="max bond dim")
    ax3 = Axis(fig[3, 1], xlabel=L"t", ylabel="truncation error")

    linkxaxes!(ax1, ax2, ax3)

    for p in ps
        grp = file["Z_$p"]
        overlaps = read(grp, "overlaps")
        nt = read(grp, "nt")
        tsp = ts[1 : nt]
        
        if p == 1
            lines!(ax1, tsp, overlaps, label=L"| 0 \rangle")
        elseif p == 2
            lines!(ax1, tsp, overlaps, label=L"| \mathbb{Z}_2 \rangle")

            maxbonds = read(grp, "maxbonds")
            truncerrs = read(grp, "truncerrs")
           
            lines!(ax2, tsp, maxbonds)
            lines!(ax3, tsp, truncerrs)
        else
            lines!(ax1, tsp, overlaps, label=L"| \mathbb{Z}_{%$p} \rangle")
        end
    end

    axislegend(ax1, position=:rt, framevisible=false)

    hidexdecorations!(ax1, ticks=false)
    hidexdecorations!(ax2, ticks=false)

    rowsize!(fig.layout, 1, Auto(2))
    rowsize!(fig.layout, 2, Auto(1))
    rowsize!(fig.layout, 3, Auto(1))

    rowgap!(fig.layout, 0)


    close(file)
    display(fig)
    
end