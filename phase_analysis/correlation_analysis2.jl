using MKL
using Interpolations
using Plots, LaTeXStrings
using HDF5

let 
    L1, dL, L2 = 8, 4, 40
    nprob = 101

    L0, p0 = 40, 0.8

    file = h5open("data/entrcorr2_avg_L$(L1)_$(dL)_$(L2)_$(nprob)x1.h5", "r")
    ps = read(file, "params/ps")
    ηs = read(file, "params/ηs")
    Ls = read(file, "params/Ls")

    grp = file["L_$(L0)"]
    corr_means = read(grp, "corr_means")
    corr_sems = read(grp, "corr_stds")

    close(file)

    idx_p = findfirst(p -> p==p0, ps)

    corr_val = corr_means[:, idx_p]
    corr_sem = corr_sems[:, idx_p] / sqrt(8L0)

    plot(0:(L0-1), corr_val, yerror=corr_sem)

end