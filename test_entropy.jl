using ITensors, ITensorMPS

include("entanglement_entropies.jl")

let 
    ss = siteinds("S=1/2", 10)
    psi = random_mps(ComplexF64, ss; linkdims=4)
    snb = Renyi_entropy(psi, 5, 2)
    snr = Renyi_entropy_region(psi, [6,7,8,9,10], 2)
    inn = mutual_information_region(psi, [1,2], [6,5], 2)
    println("Von Neumann entropy (bipart): ", snb)
    println("Von Neumann entropy (region): ", snr)
    println("Mutual information (region): ", inn)
end