using HDF5
using Interpolations
using Optim
include("entropy_calc.jl")

function object_function(data, Ls, ps, pc::Real, nu::Real, eta::Real; numsamp = 10)
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