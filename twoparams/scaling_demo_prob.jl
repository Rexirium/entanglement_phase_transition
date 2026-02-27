using MKL
using HDF5
using FiniteSizeScaling
using Interpolations
using Plots, LaTeXStrings

let 
    # Parameters
    L1, dL, L2 = 4, 2, 18
    nprob, neta = 21, 21

    η0 = 0.0
    file = h5open("data/entropy_data_L$(L1)_$(dL)_$(L2)_$(nprob)x$(neta).h5", "r")
    type_str = read(file, "datatype")
    grp = file["params"]
    Ls = read(grp, "Ls")
    ps = read(grp, "ps")
    ηs = read(grp, "ηs")

    type = eval(Meta.parse(type_str))

    nL = length(Ls)
    ps_knots = range(ps[1], ps[end], length(ps))
    idx_η0 = findfirst(ηs .== η0)

    data_with_err = []
    fit_weights = []
    interps = []
    for l in Ls
        means = read(file, "L=$l/means")[:, idx_η0]
        sems = read(file, "L=$l/stds")[:, idx_η0]
        push!(data_with_err, [ps, means, sems, l])
        push!(fit_weights, 1.0 ./ (sems .^ 2))
        push!(interps, cubic_spline_interpolation(ps_knots, means))
    end
    close(file)

    # Finite-size scaling analysis
    x_scaled(X, L, v1, v2) = L^(1/v2) .* (X .- v1)
    y_scaled(Y, L, v1, v2) = Y .- interps[findfirst(Ls .== L)](v1)

    scaled_data, residuals, min_res, best_pc, best_nu = fss_two_var(
        data = data_with_err,
        xs = x_scaled, ys = y_scaled,
        v1i=0.0, v1f = 1.0, n1 = 100,
        v2i = 0.5, v2f = 5.0, n2 = 100,
        p = 6,
        #weights = fit_weights,
        norm_y = false
    )
    
    plot_data(data_with_err, 
        xlabel=L"(p - p_c) L^{1/\nu}",
        ylabel=L"S_1(p) - S_1(p_c)",
        legend=:best, 
        xguidefontsize=12,
        yguidefontsize=12, 
        size=(600,400)
        )
    
    #=
    plot_contour(residuals, 
        v1i=0.0, v1f=1.0, n1 = 100,
        v2i=0.5, v2f=5.0, n2=100,
        xlabel=L"p_c", ylabel=L"\nu",
        levels=20
    )
    =#
end