using MKL
using Interpolations
using Plots, LaTeXStrings
using HDF5

let 
    L1, dL, L2 = 8, 4, 40
    nprob, neta = 21, 20

    L0, p0, η0 = 40, 0.3, 0.25

    file = h5open("data/entrcorr_avg_L$(L1)_$(dL)_$(L2)_$(nprob)x$(neta).h5", "r")
    ps = read(file, "params/ps")
    ηs = read(file, "params/ηs")
    Ls = read(file, "params/Ls")

    grp = file["L_$(L0)"]
    corr_means = read(grp, "corr_means")
    corr_stds = read(grp, "corr_stds")

    close(file)

    idx_p = findfirst(p -> p==p0, ps)
    idx_η = findfirst(η -> η==η0, ηs)

    corr_val = corr_means[:, idx_p, idx_η]
    corr_std = corr_stds[:, idx_p, idx_η]

    plot(0:(L0-1), corr_val, yerror=corr_std)

end