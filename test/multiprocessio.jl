using Distributed
using HDF5

if nprocs() == 1
    addprocs(8)  # add worker processes if not already added
end

#=
const xs = 0.1:0.1:2.0
const ys = 0.1:0.1:2.0
const xy = vec([(x, y) for x in xs, y in ys])
@everywhere const xys = $xy
=#
@everywhere function worker_task(xy)
    # Simulate some work and return a result
    sleep(0.01)  # simulate time-consuming task
    x, y = xy
    return (x + y) * rand()
end

let 
    
    xs = 0.1:0.1:2.0
    ys = 0.1:0.1:2.0
    xys = vec([(x, y) for x in xs, y in ys])
    
    n = length(xys)

    @time begin
        if myid() == 1
            h5open("test/test_multiprocessio.h5", "w") do file
                write(file, "name", "hello")
                #dset = create_dataset(file, "data", datatype(Float64), dataspace(20, 20, 10))
            end
        end
        for i in 1:10
            data = pmap(xy -> worker_task(xy), xys)  # run tasks in parallel

            if myid() == 1    
                h5open("test/test_multiprocessio.h5", "r+") do file
                    grp = create_group(file, "iteration_$i")
                    write(grp, "iteration", reshape(data, 20, 20))
                end
            end
                
                #dset[:, :, i] = reshape(data, 20, 20)  # write results to HDF5 dataset
        end    
        
    end
end