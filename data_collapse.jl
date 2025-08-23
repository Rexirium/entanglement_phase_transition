using HDF5
include("entropy_calc.jl")

function object_function(data, Ls, ps, pc::Real, nu::Real, eta::Real)
    critical_entropies = [ entropy_mean(l, 4l, pc, eta, l ÷ 2; numsamp=10) for l in Ls ]
    ys = data .- transpose(critical_entropies)
    xs = (ps .- pc) .* transpose(Ls .^(1/nu))
    
end