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
    file = h5open("time_evolve_data.h5", "r")
    ps = read(file, "ps")
    ηs = read(file, "ηs")
    T = read(file, "T")
    L = read(file, "L")
    prob_evolves = read(file, "prob_evolves")
    prob_distris = read(file, "prob_distris")
    eta_evolves = read(file, "eta_evolves")
    eta_distris = read(file, "eta_distris")
    close(file)
     # Plotting
     # Entanglement entropy time evolution for varying p
     pt = plot(0:T, prob_evolves, lw = 2,
         xlabel=L"t", ylabel=L"S_1", 
         title="entanglement entropy evolution for varying p",
         label=string.(collect(ps)'),
         legend_title=L"p")
     # Entanglement entropy distribution for varying p at final time
     px = plot(0:L, prob_distris, lw=2,
         xlabel=L"x", ylabel=L"S_1", 
         title="entanglement entropy distribution for varying p",
         label=string.(collect(ps)'),
         legend_title=L"p")
     # Entanglement entropy time evolution for varying η
     et = plot(0:T, eta_evolves, lw=2,
         xlabel=L"t", ylabel=L"S_1",
         title="entanglement entropy evolution for varying η",
         label=string.(collect(ηs)'),
         legend_title=L"\eta")
     # Entanglement entropy distribution for varying η at final time
     ex = plot(0:L, eta_distris, lw=2,
         xlabel=L"x", ylabel=L"S_1",
         title="entanglement entropy distribution for varying η",
         label=string.(collect(ηs)'),
         legend_title=L"\eta")

    plot(pt, px, et, ex, layout=(2,2), size=(1200, 800))

end