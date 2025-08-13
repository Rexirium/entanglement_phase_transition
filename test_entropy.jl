using ITensors, ITensorMPS

include("entanglement_entropies.jl")

let 
    ss = siteinds("S=1/2", 10)
    psi = random_mps(ss; linkdims=4)
    svn1 = von_Neumann_entropy(psi, 9)
    svn2 = von_Neumann_entropy(psi, 8)
    svns = von_Neumann_entropy_single(psi, 9)
    println("Von Neumann entropy (MPS): ", svn1+svn2)
    println("Von Neumann entropy (single site): ", svns)
end