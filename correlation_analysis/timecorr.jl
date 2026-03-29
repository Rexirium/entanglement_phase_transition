using MKL
using HDF5
using CairoMakie

let 
    L1, dL, L2 = 6, 2, 20
    nprob, neta = 51, 1

    ################ reading data ################
    #============================================#
    file = h5open("data/nhcnot_entrcorr_avg_L$(L1)_$(dL)_$(L2)_$(nprob)x$(neta).h5", "r")
    type_str = read(file, "datatype")
    ps = read(file, "params/ps")
    η0 = read(file, "params/ηs")[1]
    Ls = read(file, "params/Ls")

    type = eval(Meta.parse(type_str))

    nL = length(Ls)
    L0 = 12
    Lidx = findfirst(==(L0), Ls)

    plist = [0.16, 0.20, 0.24, 0.28]
    pidxs = indexin(plist, ps)
    nplist = length(plist)
    
    grp = file["results"]
    entr_means = read(grp, "entr_means")
    entr_logms = read(grp, "entr_logms")
    entr_sems = read(grp, "entr_sems")
    truncerrs = read(grp, "truncerrs")
    
    dset1 = grp["corr_means"]
    dset2 = grp["corr_sems"]

    corr_means = dset1[:, :, Lidx]
    corr_sems = dset2[:, :, Lidx]
    timecorrs = read(grp, "timecorrs")[:, :, Lidx]
    close(file)

    fig = Figure(size=(600, 800))

    ax1 = Axis(fig[1, 1], title="Entanglement Entropy", xlabel=L"p", ylabel=L"S(L/2)")
    lines!(ax1, ps, entr_means[:, Lidx], label="mean")
    lines!(ax1, ps, entr_logms[:, Lidx], label="log mean")
    axislegend(ax1)

    ts = 0 : 4L0-1
    ax2 = Axis(fig[2, 1], title="Time Correlation", xlabel=L"\tau", ylabel=L"C(\tau)")
    for i in 1:nplist
        s = pidxs[i]
        lines!(ax2, ts, timecorrs[1 : 4L0, s], label=L"p=%$(plist[i])")
    end
    axislegend(ax2)

    display(fig)
end