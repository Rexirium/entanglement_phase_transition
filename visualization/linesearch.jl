using MKL, LinearAlgebra
using HDF5, Interpolations
using CairoMakie

include("../linear_regress.jl")

let 
    L1, dL, L2 = 10, 2, 40
    nprob, neta = 21, 21

    ################ reading data ################
    #============================================#
    file = h5open("data/nh_entrcorr_avg_L$(L1)_$(dL)_$(L2)_$(nprob)x$(neta).h5", "r")
    type_str = read(file, "datatype")
    ps = read(file, "params/ps")
    ηs = read(file, "params/ηs")
    Ls = read(file, "params/Ls")

    type = eval(Meta.parse(type_str))

    nL = length(Ls)
    L0 = 32
    L_list = [10, 16, 20, 24, 30, 36, 40]
    Lidx = findfirst(==(L0), Ls)
    Lidcs = indexin(L_list, Ls)
    nL_list = length(L_list)
    
    grp = file["results"]
    entr_means = read(grp, "entr_means")
    entr_sems = read(grp, "entr_sems")
    truncerrs = read(grp, "truncerrs")
    
    dset1 = grp["corr_means"]
    dset2 = grp["corr_sems"]

    corr_means = dset1[:, :, :, Lidx]
    corr_sems = dset2[:, :, :, Lidx]

    close(file)

    # define the path
    pathx = reverse(ps[nprob ÷ 2 + 1: end])
    pathy = ηs[1: neta ÷ 2 + 1]
    pathlen = length(pathx)
    points = Tuple{type, type}[]
    
    truncerr = Matrix{type}(undef, pathlen, nL_list)
    entr_data = Matrix{type}(undef, pathlen, nL_list)
    entr_err = Matrix{type}(undef, pathlen, nL_list)

    corr_data = Vector{type}[]
    corr_err = Vector{type}[]

    for s in 1:pathlen
        entr_data[s, :] .= entr_means[nprob + 1 - s, s, Lidcs]
        entr_err[s, :] .= entr_sems[nprob + 1 - s, s, Lidcs]
        truncerr[s, :] .= truncerrs[nprob + 1 - s, s, Lidcs]

        if isodd(s)
            push!(corr_data, corr_means[1:L0, nprob + 1 - s, s])
            push!(corr_err, corr_sems[1:L0, nprob + 1 - s, s])
            push!(points, (pathx[s], pathy[s]))
        end
    end
    npt = length(points)

    # calculate the entropy indices at different p and η
    indices = zeros(type, nprob, neta)
    for i in 1:nprob
        for j in 1:neta
            xs = log.(Ls .+ 1e-10)
            ys = log.(abs.(entr_means[i, j, :] .+ 1e-10))
            yerrs = abs.(entr_sems[i, j, :] ./ entr_means[i, j, :])
            index = linregress(xs, ys)
            
            if (abs(index) > 0.25 && j <= 1) 
                indices[i, j] = 0.0
            else
                indices[i, j] = index
            end
            #indices[i, j] = index
        end
    end

    ############## Visualize ##################
    #=========================================#
    sizetheme = Theme(Axis = (
        titlesize=18, 
        xlabelsize=18, 
        ylabelsize=18
    ))

    fontsizetheme = merge(sizetheme, theme_latexfonts())

    set_theme!(fontsizetheme)

    fig = Figure(backgroundcolor = RGBf(0.98, 0.98, 0.98),
        size=(800, 800))
    ga = fig[1, 1] = GridLayout()

    #Heatmap for entropy index
    ax1 = Axis(ga[1, 1], 
        xlabel=L"p", 
        ylabel=L"\eta", 
        xticks=(0.0:0.25:1.0), 
        yticks=(0.0:0.3:0.9), 
        title="Entropy scale index", 
    )

    hm = heatmap!(ax1, ps, ηs, indices, 
        colormap=:plasma, 
        colorrange=(-0.1, 0.7)
    )
    lines!(ax1, pathx, pathy, linewidth = 2, color=:red)
    scatter!(ax1, points, color=1:npt, marker=:diamond, markersize=12, colormap=:tab10)
    Colorbar(ga[1, 2], hm, ticks=0.0:0.2:0.8)

    # error line for entropy alone the path in the heatmap
    ax2 = Axis(fig[1, 2], 
        xlabel=L"p", 
        ylabel=L"S_\mathrm{vN}", 
        title="Entropy changes",
    )

    for (i, l) in enumerate(L_list)
        lines!(ax2, pathx, entr_data[:, i], linewidth=2, label=L"L = %$(l)")
        errorbars!(ax2, pathx, entr_data[:, i], entr_err[:, i], 
            whiskerwidth=10, linewidth=1.5)
    end
    axislegend(ax2)

    # total truncation error
    cg = cgrad(:viridis, nL_list, categorical=true)
    ax3 = Axis(fig[2, 1], 
        xlabel=L"p", 
        ylabel=L"\epsilon", 
        limits=(nothing, nothing , 1e-18, 1e-1), 
        yscale = log10,
        title="Total truncation error",
    )
    for (i, l) in enumerate(L_list)
        lines!(ax3, pathx, truncerr[:, i], color = cg[i], label=L"L = %$(l)")
    end
    axislegend(ax3)

    ax4 = Axis(fig[2, 2], 
        xlabel=L"r", 
        ylabel=L"C_z(r)", 
        title="Correlation function at L = $L0", 
        titlefont=:bold    
    )

    for (i, pt) in enumerate(points)
        lines!(ax4, 0 : L0-1, corr_data[i], linewidth=1.5, label=L"p = %$(pt[1]),\; \eta = %$(pt[2]) ")
        errorbars!(ax4, 0 : L0-1, corr_data[i], corr_err[i], 
            whiskerwidth=10, linewidth=1)
    end
    axislegend(ax4)
    
    rowgap!(fig.layout, 5)
    colgap!(fig.layout, 5)
    fig
    #save("figures/nh_visual.png", fig)
end
