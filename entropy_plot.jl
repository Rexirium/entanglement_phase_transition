using Plots, LaTeXStrings
using HDF5
default(
    grid=false, 
    titlelocation=:left,
    titlefontsize=14,
    framestyle=:box,
    legend=:topright,
    guidefontsize=12,
    legendfontsize=10,
    tickfontsize=10,
    bottommargin=1.5Plots.mm,
    leftmargin=3Plots.mm
)

let 
    # Read data from HDF5 file
    prob_scales_mean = Matrix{Float64}(undef, nprob, nL)
    prob_scales_std = Matrix{Float64}(undef, nprob, nL)
    eta_scales_mean = Matrix{Float64}(undef, neta, nL)
    eta_scales_std = Matrix{Float64}(undef, neta, nL)

    file = h5open("entropy_scale_data.h5", "r")

    Ls = read(file, "params/Ls")
    ps = read(file, "params/ps")
    ηs = read(file, "params/ηs")
    p0 = read(file, "params/p0")
    η0 = read(file, "params/η0")

    for (i,l) in enumerate(Ls)
        prob_scales_mean[:, i] .= read(file, "results_L=$l/prob_scales_mean")
        prob_scales_std[:, i] .= read(file, "results_L=$l/prob_scales_std")
        eta_scales_mean[:, i] .= read(file, "results_L=$l/eta_scales_mean")
        eta_scales_std[:, i] .= read(file, "results_L=$l/eta_scales_std")
    end
    close(file)
    # Plotting
    # Entanglement entropy scaling for varying p
    pp = plot(ps, prob_scales_mean,
         yerror=prob_scales_std, lw=1,
         xlabel=L"p", ylabel=L"S_1", 
         title="Entanglement entropy for varying p, \eta=$η0",
         label=string.(Ls'),
         legend_title=L"L")
    # Entanglement entropy scaling for varying η
    ep = plot(ηs, eta_scales_mean, 
         yerror=eta_scales_std, lw=1,
         xlabel=L"\eta", ylabel=L"S_1",
         title="Entanglement entropy for varying η, p=$p0",
         label=string.(Ls'),
         legend_title=L"L")

    plot(pp, ep, layout=(1,2), size=(1000, 600), dpi=1200)
    savefig("figures/entropy_plot_sampx100.png")
end

