using MKL
using LinearAlgebra
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
            xs = log.(collect(Ls))
            ys = log.(abs.(entropy_datas[i, j, :]))
            yerrs = abs.(entropy_error[i, j, :] ./ entropy_datas[i, j, :])
            indices[i, j] = linregress(xs, ys, yerrs)
        end
    end
#=
    heatmap(ps, ηs, indices',
            xlabel=L"p", ylabel=L"\eta",
            title="Entropy scaling index",
            colorbar_title="index")
=#
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
end