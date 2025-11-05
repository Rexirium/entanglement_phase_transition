using Plots, LaTeXStrings
using HDF5

default(    
    grid=false, 
    titlelocation=:center,
    framestyle=:box,
    legend=:topright,
    guidefontsize=14,
    legendfontsize=10,
    tickfontsize=10,
    bottommargin=1Plots.mm,
    leftmargin=2Plots.mm
)

let 
    nprob, neta = 11, 11
    file = h5open("data/critical_params_$(nprob)x$(neta).h5", "r")
    ηs = read(file, "range/ηs")
    ps = read(file, "range/ps")

    grp = file["critical"]
    prob_crit = read(grp, "prob_crit")
    nu_prob = read(grp, "nu_prob")
    eta_crit = read(grp, "eta_crit")
    nu_eta = read(grp, "nu_eta")
    close(file)

    plot(prob_crit, ηs, marker=:o, lw=2,
         ylabel=L"\eta", xlabel=L"p",
         title="2D phase diagram", 
         label=L"p_c")
    plot!(ps, eta_crit, marker=:s, lw=2,
          label=L"\eta_c")
    #savefig("figures/phase_diagram_$(nprob)x$(neta).png")
end