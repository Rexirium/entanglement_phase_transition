using HDF5
using Interpolations
using Optim
include("entropy_calc.jl")

function object_function(pc::Real, nu::Real, eta::Real, data, Ls, ps; numsamp = 10)
    """
    Objective function for data collapse.
    """
    critical_entropies = [ entropy_mean(l, 4l, pc, eta, l ÷ 2; numsamp=numsamp) for l in Ls ]
    ys = data .- transpose(critical_entropies)
    xs = (ps .- pc) .* transpose(Ls .^(1/nu))

    nL = length(Ls)
    itps = [cubic_spline_interpolation(xs[:,k], ys[:,k]) for k in 1:nL]
    xi =  sort(union(xs...))

    Rsq = 0.0
    for x in xi
        yis = Float64[]
        for k in 1:nL
            if x < bounds(itps[k])[1][1] || x > bounds(itps[k])[1][2]
                continue
            end
            push!(yis, itps[k](x))
        end
        Rsq += var(yis, corrected=false) * length(yis) 
    end
    return Rsq
end

function data_collapse(datas, Ls, ps, ηs, p0=0.5, nu0=1.0; numsamp=10)
    """
    Perform data collapse to find the optimal critical point and critical exponent.
    """
    neta = length(ηs)
    critical_params = []
    for j in 1:neta
        data = datas[:,:,j]
        η = ηs[j]
        obj(pc_nu) = object_function(pc_nu[1], pc_nu[2], η, data, Ls, ps; numsamp=numsamp)
        res = optimize(obj, [p0, nu0], GradientDescent(), Optim.Options(g_tol=1e-6, iterations=1000))
        push!(critical_params, Optim.minimizer(res))
    end
    return hcat(critical_params...)'
end