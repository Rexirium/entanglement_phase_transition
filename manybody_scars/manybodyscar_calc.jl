using Distributed
using HDF5

if nworkers() == 0
    addprocs(4)
end

@everywhere include("manybodyscar.jl")
@everywhere function main(lsize::Int, period::Int, freq::Int)
    ss = siteinds("S=1/2", lsize)
    psi = make_initialstate(ss, period, "Up")
    
    obs = MyObserver(freq)
    tebd_pxp!(psi, 30.0, 600, obs; maxdim=400, cutoff=1e-14)
    return obs
end

let 
    L = 30
    ps = 1 : 4
    q = 2

    results = pmap( p -> main(L, p, q), ps)

    h5open("manybody_scars/pxp_L$(L).h5", "w") do file
        write(file, "params/nsteps", (600 ÷ q + 1))

        for (i, res) in enumerate(results)
            grp = create_group(file, "Z_$(ps[i])")
            write(grp, "entropies", res.entropies)
            write(grp, "correlations", res.correlations)
            write(grp, "overlaps", res.overlaps)
            write(grp, "maxbonds", res.maxbonds)
            write(grp, "truncerrs", res.truncerrs)
            close(grp)
        end
    end
end