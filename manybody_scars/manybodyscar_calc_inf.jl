using Distributed
using HDF5

if nworkers() == 0
    addprocs(5)
end

@everywhere include("manybodyscar.jl")
@everywhere begin
    MKL.set_num_threads(1)
    ITensors.BLAS.set_num_threads(1)
    ITensors.Strided.set_num_threads(1)
end
@everywhere function main(period::Int, freq::Int)
    len_uc = lcm(period, 3)
    ss = siteinds("S=1/2", len_uc)
    ipsi = make_initialinfstate(ss, period, "Up")
    
    obs = MyObserver(freq)
    tebd_pxp!(ipsi, 30.0, 600, obs; maxdim=400, cutoff=1e-12, etol=1e-2)
    println("period $period initial state time evolution finished!")
    return obs
end

let 
    ps = 1 : 4
    q = 2

    results = pmap( p -> main(p, q), ps)

    h5open("manybody_scars/pxp_inf.h5", "w") do file
        grp = create_group(file, "params")
        write(grp, "nsteps", (600 ÷ q + 1))
        write(grp, "periods", collect(ps))

        for (i, res) in enumerate(results)
            grp = create_group(file, "Z_$(ps[i])")
            write(grp, "entropies", res.entropies)
            write(grp, "correlations", res.correlations)
            write(grp, "overlaps", res.overlaps)
            write(grp, "maxbonds", res.maxbonds)
            write(grp, "truncerrs", res.truncerrs)
            write(grp, "nt", length(res.entropies))
            close(grp)
        end
    end
end