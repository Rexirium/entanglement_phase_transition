using ITensors, ITensorMPS

include("entanglement_entropies.jl")

let 
    ss = siteinds("S=1/2", 10)
    psi = random_mps(ComplexF64, ss; linkdims=4)
    svnb = von_Neumann_entropy(psi, 5)
    svnr = von_Neumann_entropy_region(psi, [6,7,8,9,10])
    ivn = mutual_information_region(psi, [1,2], [6,5], 1)
    println("Von Neumann entropy (bipart): ", svnb)
    println("Von Neumann entropy (region): ", svnr)
    println("Mutual information (region): ", ivn)
end