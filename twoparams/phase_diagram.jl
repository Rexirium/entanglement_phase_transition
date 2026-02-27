using MKL
using HDF5
using Interpolations
using FiniteSizeScaling

let 
    L1, dL, L2 = 6, 2, 18
    nprob, neta = 21, 21
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
        @views entropy_datas[:, :, i] .= read(file, "L=$l/means")
        @views entropy_error[:, :, i] .= read(file, "L=$l/stds")
    end
    close(file)

    ps_knots = range(ps[1], ps[end], length(ps))
    prob_crit = type[]
    nu_prob = type[]
    for (j, η) in enumerate(ηs)
        data_with_err = []
        fit_weights = []
        interps = []
        for (k, l) in enumerate(Ls)
            means = entropy_datas[:, j, k]
            sems = entropy_error[:, j, k]
            push!(data_with_err, [ps, means, sems, l])
            push!(fit_weights, 1.0 ./ (sems .^ 2))
            push!(interps, cubic_spline_interpolation(ps_knots, means))
        end

        x_scaled(X, L, v1, v2) = L^(1/v2) .* (X .- v1)
        y_scaled(Y, L, v1, v2) = Y .- interps[findfirst(Ls .== L)](v1)

        println("critical prob for η = $η, v1 = pc, v2 = nu")
        scaled_data, residuals, min_res, best_pc, best_nu = fss_two_var(
            data = data_with_err,
            xs = x_scaled, ys = y_scaled,
            v1i=0.0, v1f = 1.0, n1 = npc,
            v2i = 0.5, v2f = 5.0, n2 = nnu,
            p = 6,
            #weights = fit_weights,
            norm_y = false
        )

        push!(prob_crit, best_pc)
        push!(nu_prob, best_nu)
        
        interps = nothing
        data_with_err = nothing
        fit_weights = nothing
    end

    ηs_knots = range(ηs[1], ηs[end], length(ηs))
    eta_crit = type[]
    nu_eta = type[]
    for (i, p) in enumerate(ps)
        data_with_err = []
        fit_weights = []
        interps = []
        for (k, l) in enumerate(Ls)
            means = entropy_datas[i, :, k]
            sems = entropy_error[i, :, k]
            push!(data_with_err, [ηs, means, sems, l])
            push!(fit_weights, 1.0 ./ (sems .^ 2))
            push!(interps, cubic_spline_interpolation(ηs_knots, means))
        end

        x_scaled(X, L, v1, v2) = L^(1/v2) .* (X .- v1)
        y_scaled(Y, L, v1, v2) = Y .- interps[findfirst(Ls .== L)](v1)

        println("critical eta for p = $p, v1 = ηc, v2 = nu")
        scaled_data, residuals, min_res, best_ηc, best_nu = fss_two_var(
            data = data_with_err,
            xs = x_scaled, ys = y_scaled,
            v1i=0.0, v1f = 1.0, n1 = npc,
            v2i = 0.5, v2f = 5.0, n2 = nnu,
            p = 6,
            #weights = fit_weights,
            norm_y = false
        )

        push!(eta_crit, best_ηc)
        push!(nu_eta, best_nu)
        
        interps = nothing
        data_with_err = nothing
        fit_weights = nothing
    end
    
    h5open("data/critical_params_$(nprob)x$(neta).h5", "w") do file
        write(file, "datatype", string(type))
        write(file, "range/ps", ps)
        write(file, "range/ηs", ηs)

        grp = create_group(file, "critical")
        write(grp, "prob_crit", prob_crit)
        write(grp, "nu_prob", nu_prob)
        write(grp, "eta_crit", eta_crit)
        write(grp, "nu_eta", nu_eta)
    end

end
    