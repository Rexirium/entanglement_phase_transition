using Distributed
using HDF5

addprocs(8)  # add worker processes

const xs = 0.1:0.1:10.0
const ys = 0.1:0.1:10.0
const xy = vec([(x, y) for x in xs, y in ys])

@everywhere const xys = $xy

@everywhere function worker_task(idx)
    # Simulate some work and return a result
    sleep(0.001)  # simulate time-consuming task
    x, y = xys[idx]
    return (x + y) * rand()
end

let 
    n = length(xys)
    @time begin
        h5open("test/test_multiprocessio.h5", "w") do file
            write(file, "name", "hello")
            dset = create_dataset(file, "data", datatype(Float64), dataspace(100, 100, 10),chunk=(100, 100, 1))

            for i in 1:10
                data = pmap(idx -> worker_task(idx), 1:n)  # run tasks in parallel
                dset[:, :, i] = reshape(data, 100, 100)  # write results to HDF5 dataset        
                
            end
        end
    end
end