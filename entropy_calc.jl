include("time_evolution.jl")

function entropy_sample(lsize::Int, ttotal::Int, prob::Real, eta::Real, b::Int=lsize ÷ 2, which_ent::Real=1; 
    cutoff::Real=1e-14, ent_cutoff::Real=1e-12)
    """
    Calculate the final entanglement entropy of the MPS after time evolution.
    """
    ss = siteinds("S=1/2", lsize)
    psi0 = MPS(ss, "Up")
    psi = time_evolve(psi0, ttotal, prob, eta, b, which_ent; cutoff=cutoff, ent_cutoff=ent_cutoff)
    return Renyi_entropy(psi, b, which_ent; cutoff=ent_cutoff)
end