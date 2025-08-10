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

    pt = plot(0:T, prob_evolves, lw = 2,
         xlabel=L"t", ylabel=L"S_1", 
         title="entanglement entropy evolution for varying p",
         label=string.(collect(ps)'),
         legend_title=L"p")
    
    px = plot(0:L, prob_distris, lw=2,
         xlabel=L"x", ylabel=L"S_1", 
         title="entanglement entropy distribution for varying p",
         label=string.(collect(ps)'),
         legend_title=L"p")

    et = plot(0:T, eta_evolves, lw=2,
         xlabel=L"t", ylabel=L"S_1",
         title="entanglement entropy evolution for varying η",
         label=string.(collect(ηs)'),
         legend_title=L"\eta")

    ex = plot(0:L, eta_distris, lw=2,
         xlabel=L"x", ylabel=L"S_1",
         title="entanglement entropy distribution for varying η",
         label=string.(collect(ηs)'),
         legend_title=L"\eta")

    plot(pt, px, et, ex, layout=(2,2), size=(1200, 800))

end