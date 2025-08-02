using Plots
include("time_evolution.jl")

let 
    L = 10
    T, b = 4L, L ÷ 2
    ps = 0.0:0.2:1.0
    ηs = 0.0:0.5:2.0
    numsamp = 10

    ss = siteinds("S=1/2", L)
    psi0 = MPS(ss, "Up")

    prob_evolves = []
    prob_distris = []

    for p in ps
        evolvesamp = []
        distrisamp = []
        for _ in 1:numsamp
            psi, evolve = entropy_evolve(psi0, T, p, 0.5, b, 1)
            distri = [Renyi_entropy(psi, x, 1) for x in 0:L]
            push!(evolvesamp, evolve)
            push!(distrisamp, distri)
        end

        meanevolve = sum(evolvesamp)/numsamp
        meandistri = sum(distrisamp)/numsamp

        push!(prob_evolves, meanevolve)
        push!(prob_distris, meandistri)
    end
    prob_evolves = hcat(prob_evolves...)
    prob_distris = hcat(prob_distris...)

    eta_evolves = []
    eta_distris = []

    for η in ηs
        evolvesamp = []
        distrisamp = []
        for _ in 1:numsamp
            psi, evolve = entropy_evolve(psi0, T, 0.5, η, b, 1)
            distri = [Renyi_entropy(psi, x, 1) for x in 0:L]
            push!(evolvesamp, evolve)
            push!(distrisamp, distri)
        end

        meanevolve = sum(evolvesamp)/numsamp
        meandistri = sum(distrisamp)/numsamp

        push!(eta_evolves, meanevolve)
        push!(eta_distris, meandistri)
    end
    eta_evolves = hcat(eta_evolves...)
    eta_distris = hcat(eta_distris...) 

    pt = plot(0:T, prob_evolves, lw = 2,
         xlabel="time", ylabel="entanglement entropy", 
         title="entanglement entropy evolution for varying p",
         label=string.(collect(ps)'), legend=:topright, framestyle=:box)
    
    px = plot(0:L, prob_distris, lw=2,
         xlabel="bipartition", ylabel="entanglement entropy", 
         title="entanglement entropy distribution for varying p",
         label=string.(collect(ps)'), legend=:topright, framestyle=:box)

    et = plot(0:T, eta_evolves, lw=2,
         xlabel="time", ylabel="entanglement entropy",
         title="entanglement entropy evolution for varying η",
         label=string.(collect(ηs)'), legend=:topright, framestyle=:box)

    ex = plot(0:L, eta_distris, lw=2,
         xlabel="bipartition", ylabel="entanglement entropy",
         title="entanglement entropy distribution for varying η",
         label=string.(collect(ηs)'), legend=:topright, framestyle=:box)

    plot(pt, px, et, ex, layout=(2,2), size=(1200, 800))

end

