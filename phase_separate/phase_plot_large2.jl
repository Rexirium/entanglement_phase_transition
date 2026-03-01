using MKL
using LinearAlgebra
using Interpolations
using Plots, LaTeXStrings
using HDF5

include("linear_regress.jl")

let
    L1, dL, L2 = 8, 4, 40
    nprob = 101

    file = h5open("data/entrcorr2_avg_L$(L1)_$(dL)_$(L2)_$(nprob)x1.h5", "r")
    type_str = read(file, "datatype")
    ps = read(file, "params/ps")
    η0 = read(file, "params/ηs")[1]
    Ls = read(file, "params/Ls")

    type = eval(Meta.parse(type_str))

    nL = length(Ls)
    entropy_datas = Matrix{type}(undef, nprob, nL)
    entropy_error = Matrix{type}(undef, nprob, nL)
    for (i,l) in enumerate(Ls)
        @views entropy_datas[:, i] .= read(file, "L_$l/entr_means")
        @views entropy_error[:, i] .= read(file, "L_$l/entr_stds")
    end
    close(file)

    indices = zeros(type, nprob)

    for i in 1:nprob
        xs = log.(Ls .+ 1e-10)
        ys = log.(abs.(entropy_datas[i, :] .+ 1e-10))
        yerrs = abs.(entropy_error[i, :] ./ entropy_datas[i, :])
        index = linregress(xs, ys)
            
        if abs(index) > 0.5 
            indices[i] = 0.0
        else
            indices[i] = index
        end
            
        #indices[i] = index
    end

    p_knots = range(ps[1], ps[end], length=nprob)
    entropy_itp = cubic_spline_interpolation(p_knots, indices)

    p_fine = range(ps[1], ps[end], length=10*nprob)
    indices_fine = [entropy_itp(p) for p in p_fine]

    
    plot(ps, indices; 
            xlabel=L"p", ylabel="entropy scaling index",
            title="Entropy scaling index at η=$(η0)",
            titlefontsize=14,
            marker=:o, lw=2,
            framestyle=:box, dpi=800
        )
    #savefig("phase_separate/entropy_scaling_index_modified.png")
    
    #=
    p0, η0 = 1.0, 0.2
    pidx = findfirst(x -> x == p0, ps)
    ηidx = findfirst(x -> x == η0, ηs)
    data = abs.(entropy_datas[pidx, ηidx, :] )
    sem = abs.(entropy_error[pidx, ηidx, :]) ./ Ls
    err = abs.(sem ./ data)
    println(data)
    plot(log.(Ls), log.(data), 
        yerror=err,
        marker=:o, lw=2,
        xlabel=L"\log L", ylabel=L"\log S_1",
        title="Entropy scaling at p=0.75, η=0.0",
        label="data")
    =#
end