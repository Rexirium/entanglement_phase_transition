using MKL
using LinearAlgebra
using Interpolations
using Plots, LaTeXStrings
using HDF5

function linregress(xs, ys)
    n = length(xs)
    A = [ones(n) xs]
    coeffs = A \ ys
    return coeffs[2]
end

function linregress(xs, ys, yerrs)
    n = length(xs)
    ws = normalize(1 ./(yerrs .^ 2), 1)
    W = diagm(sqrt.(ws))
    A = [ones(n) xs]
    Aw = W * A
    yw = W * ys
    coeffs = Aw \ yw
    return coeffs[2]
end

let 
    L1, dL, L2 = 6, 2, 18
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
        entropy_datas[:, :, i] .= read(file, "L=$l/means")
        entropy_error[:, :, i] .= read(file, "L=$l/stds")
    end
    close(file)

    indices = zeros(type, nprob, neta)

    for i in 1:nprob
        for j in 1:neta
            xs = log.(Ls .+ 1)
            ys = log.(abs.(entropy_datas[i, j, :]) .+ log(2))
            yerrs = abs.(entropy_error[i, j, :] ./ entropy_datas[i, j, :])
            index = linregress(xs, ys)
            if index > 1.2 
                indices[i, j] = 1.2
            elseif index < 0 || isnan(index)
                indices[i, j] = 0
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

    
    heatmap(ps, ηs, indices',
            xlabel=L"p", ylabel=L"\eta",
            title="Entropy scaling index",
            titlefontsize=14,
            colorbar_title="index",
            framestyle=:box, dpi=800
            )
    savefig("phase_apart/entropy_scaling_index_modified.png")
    #=
    p0, η0 = 0.75, 0.0
    pidx = findfirst(x -> x == p0, ps)
    ηidx = findfirst(x -> x == η0, ηs)
    data = abs.(entropy_datas[pidx, ηidx, :])
    err = abs.(entropy_error[pidx, ηidx, :]./data)
    println(data)
    plot(log.(Ls), log.(data), 
        yerror=err,
         marker=:o, lw=2,
         xlabel=L"L", ylabel="Entropy",
         title="Entropy scaling at p=0.75, η=0.0",
         label="data")
    =#
end