using HDF5
using Base.Threads
using ITensors, ITensorMPS

include("entanglement_entropies.jl")

mylock = ReentrantLock()

let 
    num = 20
    results = zeros(num)
    @threads for i in 1:num
        ss = siteinds("S=1/2", 10)
        psi = random_mps(ComplexF64, ss; linkdims=4)
        svn = von_Neumann_entropy(psi, 5)
        @lock mylock results[i] = svn
        println("Iteration $i: Von Neumann Entropy = $svn")
    end

    h5open("test_entropy_results.h5", "w") do file
        write(file, "results", results)
    end
end