using Plots, LaTeXStrings
using HDF5

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
    file = h5open("critical_params.h5", "r")
    ηs = read(file, "ηs")
    critical_params = read(file, "critical_params")

    pcs = critical_params[:,1]
    plot(pcs, ηs, lw=2, marker=:o,
         ylabel=L"\eta", xlabel=L"p_c", 
         title="2D phase diagram")
end