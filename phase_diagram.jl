using HDF5
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
include("data_collapse.jl")

let 
    file = h5open("entropy_data.h5", "r")
    ps = read(file, "ps")
    ηs = read(file, "ηs")
    Ls = read(file, "Ls")
    entropy_datas = read(file, "entropy_datas")
    close(file)

    critical_params = data_collapse(entropy_datas, Ls, ps, ηs; numsamp=100)
    pcs = critical_params[:,1]

    plot(pcs, ηs, lw=2, marker=:o,
         ylabel=L"\eta", xlabel=L"p_c", 
         title="2D phase diagram")
end