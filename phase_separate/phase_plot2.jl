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
    L1, dL, L2 = 8, 4, 40
    nprob, neta = 21, 20

    file = h5open("data/entrcorr_avg_L$(L1)_$(dL)_$(L2)_$(nprob)x$(neta).h5", "r")
    type_str = read(file, "datatype")
    ps = read(file, "params/ps")
    ηs = read(file, "params/ηs")
    Ls = read(file, "params/Ls")

    type = eval(Meta.parse(type_str))

    nL = length(Ls)
    entropy_datas = Array{type}(undef, nprob, neta, nL)
    entropy_error = Array{type}(undef, nprob, neta, nL)
    for (i,l) in enumerate(Ls)
        @views entropy_datas[:, :, i] .= read(file, "L_$l/entr_means")
        @views entropy_error[:, :, i] .= read(file, "L_$l/entr_stds")
    end
    close(file)

    indices = zeros(type, nprob, neta)

    for i in 1:nprob
        for j in 1:neta
            xs = log.(Ls .+ 1e-10)
            ys = log.(abs.(entropy_datas[i, j, :] .+ 1e-10))
            yerrs = abs.(entropy_error[i, j, :] ./ entropy_datas[i, j, :])
            index = linregress(xs, ys)
            #=
            if (abs(index) > 1.3 && j <= 3) || isnan(index)
                indices[i, j] = 0.0
            else
                indices[i, j] = index
            end
            =#
            indices[i, j] = index
        end
    end

    p_knots = range(ps[1], ps[end], length=nprob)
    η_knots = range(ηs[1], ηs[end], length=neta)
    entropy_itp = cubic_spline_interpolation((p_knots, η_knots), indices)

    p_fine = range(ps[1], ps[end], length=10*nprob)
    η_fine = range(ηs[1], ηs[end], length=10*neta)
    indices_fine = [entropy_itp(p, η) for p in p_fine, η in η_fine]

    #=
    heatmap(ps, ηs, indices'; 
            xlabel=L"p", ylabel=L"\eta",
            title="Entropy scaling index",
            titlefontsize=14,
            colorbar_title="index",
            framestyle=:box, dpi=800
            )
    #savefig("phase_separate/entropy_scaling_index_modified.png")
    =#
    p0, η0 = 1.0, 0.2
    pidx = findfirst(x -> x == p0, ps)
    ηidx = findfirst(x -> x == η0, ηs)
    data = abs.(entropy_datas[pidx, ηidx, :] )
    stderr = abs.(entropy_error[pidx, ηidx, :]) ./ Ls
    err = abs.(stderr ./ data)
    println(data)
    plot(log.(Ls), log.(data), 
        yerror=err,
         marker=:o, lw=2,
         xlabel=L"\log L", ylabel=L"\log S_1",
         title="Entropy scaling at p=0.75, η=0.0",
         label="data")
end