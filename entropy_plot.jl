using Plots, LaTeXStrings
default(
    grid=false, 
    titlelocation=:left,
    framestyle=:box,
    legend=:topright,
    guidefontsize=14,
    legendfontsize=10,
    tickfontsize=10,
    bottommargin=2Plots.mm,
    leftmargin=4Plots.mm
)

let 
    # Read data from HDF5 file
    file = h5open("entropy_scale_data.h5", "r")
    ps = read(file, "ps")
    ηs = read(file, "ηs")
    Ls = read(file, "Ls")
    prob_scales_mean = read(file, "prob_scales_mean")
    prob_scales_std = read(file, "prob_scales_std")
    eta_scales_mean = read(file, "eta_scales_mean")
    eta_scales_std = read(file, "eta_scales_std")
    close(file)
    # Plotting
    # Entanglement entropy scaling for varying p
    pp = plot(ps, prob_scales_mean, lw=2,
         yerror=prob_scales_std,
         xlabel=L"p", ylabel=L"S_1", 
         title="entanglement entropy for varying p",
         label=string.(collect(Ls)'),
         legend_title=L"L")
    # Entanglement entropy scaling for varying η
    ep = plot(Ls, eta_scales_mean, lw=2,
         yerror=eta_scales_std,
         xlabel=L"\eta", ylabel=L"S_1",
         title="entanglement entropy for varying η",
         label=string.(collect(Ls)'),
         legend_title=L"L")

    plot(pp, ep, layout=(1,2), size=(800, 500))

end

