using MKL
using LinearAlgebra
using Interpolations
using CairoMakie
using HDF5

include("../linear_regress.jl")

let 
    L1, dL, L2 = 4, 2, 18
    nprob, neta = 21, 21

    file = h5open("data/entropy_data_L$(L1)_$(dL)_$(L2)_$(nprob)x$(neta).h5", "r")
    type_str = read(file, "datatype")
    ps = read(file, "params/ps")
    ηs = read(file, "params/ηs")
    Ls = read(file, "params/Ls")

    type = eval(Meta.parse(type_str))

    nL = length(Ls)
    entropy_datas = Array{type}(undef, nprob, neta, nL)
    entropy_error = Array{type}(undef, nprob, neta, nL)
    for (i,l) in enumerate(Ls)
        @views entropy_datas[:, :, i] .= read(file, "L=$l/means")
        @views entropy_error[:, :, i] .= read(file, "L=$l/stds")
    end
    close(file)

    indices = zeros(type, nprob, neta)

    for i in 1:nprob
        for j in 1:neta
            xs = log.(Ls)
            ys = log.(abs.(entropy_datas[i, j, :]))
            yerrs = abs.(entropy_error[i, j, :] ./ entropy_datas[i, j, :])
            index = linregress(xs, ys)
            if (abs(index) > 1.3 && j <= 3) || isnan(index)
                indices[i, j] = 0.0
            else
                indices[i, j] = index
            end
            #indices[i, j] = isnan(index) ? 0.0 : index
        end
    end

    p_knots = range(ps[1], ps[end], length=nprob)
    η_knots = range(ηs[1], ηs[end], length=neta)
    entropy_itp = cubic_spline_interpolation((p_knots, η_knots), indices)

    p_fine = range(ps[1], ps[end], length=10*nprob)
    η_fine = range(ηs[1], ηs[end], length=10*neta)
    indices_fine = [entropy_itp(p, η) for p in p_fine, η in η_fine]

    
    fig = Figure()
    ax = Axis(fig[1, 1], title="Entropy Scaling Index", 
        xlabel=L"p", xticks=(0.0:0.25:1.0), 
        ylabel=L"\eta", yticks=(0.0:0.3:0.9)
    )

    hm = heatmap!(ax, p_fine, η_fine, indices_fine, colorrange=(-0.1, 1.25), colormap=:plasma)
    Colorbar(fig[1, 2], hm, label = "Scaling Index")
    fig
    
end