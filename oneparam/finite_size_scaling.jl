using MKL
using HDF5
using FiniteSizeScaling
using Interpolations
using Plots, LaTeXStrings

let 
    # Parameters
    L1, dL, L2 = 6, 2, 18
    nprob, neta = 21, 21

    file = h5open("data/oneparam_L$(L1)_$(dL)_$(L2)_$(nprob)x$(neta).h5", "r")
    type_str = read(file, "datatype")
    grp = file["params"]
    Ls = read(grp, "Ls")
    ps = read(grp, "ps")
    ηs = read(grp, "ηs")
    p0 = read(grp, "p0")
    η0 = read(grp, "η0")

    type = eval(Meta.parse(type_str))

    nL = length(Ls)
    ps_knots = range(ps[1], ps[end], length(ps))
    data_with_err = []
    fit_weights = []
    interps = []
    for l in Ls
        means = read(file, "results_L=$l/prob_mean")
        stds = read(file, "results_L=$l/prob_std")
        push!(data_with_err, [ps, means, stds, l])
        push!(fit_weights, 1.0 ./ (stds .^ 2))
        push!(interps, cubic_spline_interpolation(ps_knots, means))
    end
    close(file)

    Sc(lsize::Int, p::Real) = interps[Ls .== lsize][1](p)

    # Finite-size scaling analysis
    x_scaled(X, L, v1, v2) = L^(v2) .* (X .- v1)
    y_scaled(Y, L, v1, v2) = Y .- Sc(L, v1)

    scaled_data, residuals, min_res, best_pc, best_nu = fss_two_var(
        data = data_with_err,
        xs = x_scaled, ys = y_scaled,
        v1i=0.0, v1f = 1.0, n1 = 100,
        v2i = 0.1, v2f = 2.0, n2 = 100,
        p = 5,
        #weights = fit_weights,
        norm_y = false
    )
    plot_data(scaled_data, 
        xlabel=L"(p - p_c) L^{1/\nu}",
        ylabel=L"S_1 - S_1(p_c)",
        legend=:best)
    #=plot_contour(residuals, 
        v1i=0.0, v1f=1.0, n1 = 100,
        v2i=1.0, v2f=5.0, n2=100,
        xlabel=L"p_c", ylabel=L"\nu",
        levels=20,
    )=#
end