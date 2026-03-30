using MKL
using LinearAlgebra
using Interpolations
using CairoMakie
using HDF5

include("../linear_regress.jl")

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
        ys = log.(abs.(entropy_datas[i, :] .+ 0.5))
        yerrs = abs.(entropy_error[i, :] ./ entropy_datas[i, :])
        index = linregress(xs, ys)
            
        if abs(index) > 0.5 
            indices[i] = 0.0
        else
            indices[i] = index
        end
            
        indices[i] = index
    end

    p_knots = range(ps[1], ps[end], length=nprob)
    entropy_itp = cubic_spline_interpolation(p_knots, indices)

    p_fine = range(ps[1], ps[end], length=10*nprob)
    indices_fine = [entropy_itp(p) for p in p_fine]

    
    fig = Figure()
    ax = Axis(fig[1, 1], title="Entropy Scaling Index", xlabel=L"p", ylabel="scaling index")
    lines!(ax, ps, indices, linewidth=2)
    fig
end