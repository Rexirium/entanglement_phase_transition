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
    L1, dL, L2 = 6, 2, 18
    file = h5open("data/entropy_scale_L$(L1)_$(dL)_$(L2).h5", "r")
    type_str = read(file, "datatype")
    Ls = read(file, "params/Ls")
    ps = read(file, "params/ps")
    ηs = read(file, "params/ηs")
    p0 = read(file, "params/p0")
    η0 = read(file, "params/η0")
    close(file)
    type = eval(Meta.parse(type_str))

    nprob, neta, nL = length(ps), length(ηs), length(Ls)

    prob_scales_mean = Matrix{type}(undef, nprob, nL)
    prob_scales_std = Matrix{type}(undef, nprob, nL)
    eta_scales_mean = Matrix{type}(undef, neta, nL)
    eta_scales_std = Matrix{type}(undef, neta, nL)

    file = h5open("data/entropy_scale_L$(L1)_$(dL)_$(L2).h5", "r")
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
         yerror=prob_scales_std, lw=1.5,
         xlabel=L"p", ylabel=L"S_1", 
         title="Entanglement entropy for varying p, \\eta=$η0",
         label=string.(Ls'),
         legend_title=L"L")
    scatter!(ps, prob_scales_mean, markersize=2, leg=false)
    # Entanglement entropy scaling for varying η
    ep = plot(ηs, eta_scales_mean, 
         yerror=eta_scales_std, lw=1.5,
         xlabel=L"\eta", ylabel=L"S_1",
         title="Entanglement entropy for varying \\eta, p=$p0",
         label=string.(Ls'),
         legend_title=L"L")
    scatter!(ηs, eta_scales_mean, markersize=2, leg=false)

    plot(pp, ep, layout=(1,2), size=(1000, 600), dpi=1200)
    savefig("figures/entropy_plot_L$(L1)_$(dL)_$(L2).png")
end

