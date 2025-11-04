using MKL
using HDF5
using Interpolations
using FiniteSizeScaling

let 
    L1, dL, L2 = 6, 2, 18
    nprob, neta = 11, 11
    npc, nnu = 200, 200

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

    ps_knots = range(ps[1], ps[end], length(ps))

    prob_critical = type[]
    nu_exponent = type[]
    for (j,η) in enumerate(ηs)
        data_with_err = []
        fit_weights = []
        interps = []
        for (k, l) in enumerate(Ls)
            means = entropy_datas[:, j, k]
            stds = entropy_error[:, j, k]
            push!(data_with_err, [ps, means, stds, l])
            push!(fit_weights, 1.0 ./ (stds .^ 2))
            push!(interps, cubic_spline_interpolation(ps_knots, means))
        end

        x_scaled(X, L, v1, v2) = L^(1/v2) .* (X .- v1)
        y_scaled(Y, L, v1, v2) = Y .- interps[findfirst(Ls .== L)](v1)

        scaled_data, residuals, min_res, best_pc, best_nu = fss_two_var(
            data = data_with_err,
            xs = x_scaled, ys = y_scaled,
            v1i=0.0, v1f = 1.0, n1 = npc,
            v2i = 0.5, v2f = 5.0, n2 = nnu,
            p = 6,
            #weights = fit_weights,
            norm_y = false
        )

        push!(prob_critical, best_pc)
        push!(nu_exponent, best_nu)
        
        interps = nothing
        data_with_err = nothing
        fit_weights = nothing
    end
    
    h5open("data/critical_params.h5", "w") do file
        write(file, "datatype", string(type))
        write(file, "ps", ps)
        write(file, "ηs", ηs)
        write(file, "p_crit", prob_critical)
        write(file, "nu_exp", nu_exponent)
    end

end
    