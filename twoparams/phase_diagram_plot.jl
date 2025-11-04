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
    file = h5open("data/critical_params.h5", "r")
    ηs = read(file, "ηs")
    p_crit = read(file, "p_crit")
    nu_exp = read(file, "nu_exp")
    close(file)

    plot(p_crit, ηs, marker=:o, lw=2,
         ylabel=L"\eta", xlabel=L"p",
         title="2D phase diagram", 
         label=L"p_c")
    #savefig("figures/phase_diagram.png")
end