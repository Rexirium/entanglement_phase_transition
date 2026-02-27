using MKL
using HDF5
using FiniteSizeScaling
using Interpolations
using Plots, LaTeXStrings

let 
    # Parameters
    L1, dL, L2 = 6, 2, 18
    nprob, neta = 21, 21

    p0 = 0.1
    file = h5open("data/entropy_data_L$(L1)_$(dL)_$(L2)_$(nprob)x$(neta).h5", "r")
    type_str = read(file, "datatype")
    grp = file["params"]
    Ls = read(grp, "Ls")
    ps = read(grp, "ps")
    ηs = read(grp, "ηs")

    type = eval(Meta.parse(type_str))

    nL = length(Ls)
    ηs_knots = range(ηs[1], ηs[end], length(ηs))
    idx_p0 = findfirst(ps .== p0)

    data_with_err = []
    fit_weights = []
    interps = []
    for l in Ls
        means = read(file, "L=$l/means")[idx_p0, :]
        sems = read(file, "L=$l/stds")[idx_p0, :]
        push!(data_with_err, [ηs, means, sems, l])
        push!(fit_weights, 1.0 ./ (sems .^ 2))
        push!(interps, cubic_spline_interpolation(ηs_knots, means))
    end
    close(file)

    # Finite-size scaling analysis
    x_scaled(X, L, v1, v2) = L^(1/v2) .* (X .- v1)
    y_scaled(Y, L, v1, v2) = Y .- interps[findfirst(Ls .== L)](v1)

    scaled_data, residuals, min_res, best_ηc, best_nu = fss_two_var(
        data = data_with_err,
        xs = x_scaled, ys = y_scaled,
        v1i=0.0, v1f = 1.0, n1 = 100,
        v2i = 0.5, v2f = 5.0, n2 = 100,
        p = 6,
        #weights = fit_weights,
        norm_y = false
    )
    
    plot_data(scaled_data, 
        xlabel=L"(\eta - \eta_c) L^{1/\nu}",
        ylabel=L"S_1(\eta) - S_1(\eta_c)",
        legend=:best, 
        xguidefontsize=12,
        yguidefontsize=12)
    
    #=
    plot_contour(residuals, 
        v1i=0.0, v1f=1.0, n1 = 100,
        v2i=0.5, v2f=5.0, n2=100,
        xlabel=L"p_c", ylabel=L"\nu",
        levels=20
    )
    =#
end