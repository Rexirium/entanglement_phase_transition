using HDF5
using Plots, LaTeXStrings

let 
    L1, dL, L2 = 4, 2, 18
    nprob, neta = 21, 21

    file = h5open("data/ent_corr_data_L$(L1)_$(dL)_$(L2)_$(nprob)x$(neta).h5", "r")
    type_str = read(file, "datatype")
    ps = read(file, "params/ps")
    ηs = read(file, "params/ηs")

    p0, η0 = 1.0, 0.0
    p_idx = findfirst(ps .== p0)
    η_idx = findfirst(ηs .== η0)

    grpc = file["L=$L2/correlation_Sz"]
    corr_func_means = read(grpc, "means")[:, p_idx, η_idx]
    corr_func_stds = read(grpc, "stds")[:, p_idx, η_idx]

    grpe = file["L=$L2/entropy_SvN"]
    entropy_mean = read(grpe, "means")[p_idx, η_idx]
    entropy_std = read(grpe, "stds")[p_idx, η_idx]
    close(file)

    println("Entropy S_vN at (p, η) = ($(p0), $(η0)) for L=$(L2): $(entropy_mean) ± $(entropy_std)")
    plot(0:(L2-1), corr_func_means, yerror=corr_func_stds,
         xlabel="Distance r", ylabel=L"C(r)",
         title="Correlation function at (p, η) = ($(p0), $(η0))",
         legend=false)
end