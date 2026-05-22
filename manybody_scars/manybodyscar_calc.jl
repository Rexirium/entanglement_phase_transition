using Distributed
using HDF5

if nworkers() == 0
    addprocs(4)
end

@everywhere include("manybodyscar.jl")

let 
    L = 18
    periods = 1 : 4

    results = pmap( p -> main(L, p), periods)

    h5open("data/pxp_L$(L).h5", "w") do file
        for (i, res) in enumerate(results)
            grp = create_group(file, "Z_$(periods[i])")
            write(grp, "entropies", res.entropies)
            write(grp, "correlations", res.correlations)
            write(grp, "overlaps", res.overlaps)
            write(grp, "maxbonds", res.maxbonds)
            write(grp, "truncerrs", res.truncerrs)
            close(grp)
        end
    end
end