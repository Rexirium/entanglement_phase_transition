using Plots, LaTeXStrings
using HDF5
default(
    grid=false, 
    titlelocation=:left,
    titlefontsize=12,
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
    L1, dL, L2 = 8, 2, 18
    nprob, neta = 21, 21

    file = h5open("data/oneparam_L$(L1)_$(dL)_$(L2)_$(nprob)x$(neta).h5", "r")
    type_str = read(file, "datatype")
    Ls = read(file, "params/Ls")
    ps = read(file, "params/ps")
    ηs = read(file, "params/ηs")
    p0 = read(file, "params/p0")
    η0 = read(file, "params/η0")

    type = eval(Meta.parse(type_str))

    nL = length(Ls)

    prob_means = Matrix{type}(undef, nprob, nL)
    prob_stds = Matrix{type}(undef, nprob, nL)
    eta_means = Matrix{type}(undef, neta, nL)
    eta_stds = Matrix{type}(undef, neta, nL)

    for (i,l) in enumerate(Ls)
        prob_means[:, i] .= read(file, "results_L=$l/prob_mean")
        prob_stds[:, i] .= read(file, "results_L=$l/prob_std")
        eta_means[:, i] .= read(file, "results_L=$l/eta_mean")
        eta_stds[:, i] .= read(file, "results_L=$l/eta_std")
    end
    close(file)
    # Plotting
    colors = palette(:darkrainbow, nL, rev=true)[:]'
    # Entanglement entropy scaling for varying p
    pp = plot(ps, prob_means, 
        yerror=prob_stds, 
        lw=1.5, lc=colors,
        xlabel=L"p", ylabel=L"S_1", 
        title="Entanglement entropy for varying p, \\eta=$η0",
        label=string.(Ls'),
        legend_title=L"L")

    # Entanglement entropy scaling for varying η
    ep = plot(ηs, eta_means, 
        yerror=eta_stds, 
        lw=1.5, lc=colors,
        xlabel=L"\eta", ylabel=L"S_1",
        title="Entanglement entropy for varying \\eta, p=$p0",
        label=string.(Ls'),
        legend_title=L"L")

    plot(pp, ep, layout=(1,2), size=(1000, 600), dpi=1200)
    #savefig("figures/entropy_plot_L$(L1)_$(dL)_$(L2)_$(nprob)x$(neta).png")
end

