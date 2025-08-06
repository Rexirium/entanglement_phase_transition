using HDF5
include("time_evolution.jl")

function entropy_sample(lsize::Int, ttotal::Int, prob::Real, eta::Real, b::Int=lsize ÷ 2, which_ent::Real=1; 
    cutoff::Real=1e-14, ent_cutoff::Real=1e-12)
    """
    Calculate the final entanglement entropy of the MPS after time evolution. (non-Hermitian case)
    """
    ss = siteinds("S=1/2", lsize)
    psi0 = MPS(ss, "Up")
    psi = mps_evolve(psi0, ttotal, prob, eta; cutoff=cutoff)
    return Renyi_entropy(psi, b, which_ent; cutoff=ent_cutoff)
end

function entropy_sample(lsize::Int, ttotal::Int, prob::Real, para::Tuple{Real, Real}, b::Int=lsize ÷ 2, which_ent::Real=1; 
    cutoff::Real=1e-14, ent_cutoff::Real=1e-12)
    """
    Calculate the final entanglement entropy of the MPS after time evolution. (weak measurement case)
    """
    ss = siteinds("S=1/2", lsize)
    psi0 = MPS(ss, "Up")
    psi = mps_evolve(psi0, ttotal, prob, para; cutoff=cutoff)
    return Renyi_entropy(psi, b, which_ent; cutoff=ent_cutoff)
end

function entropy_mean(lsize::Int, ttotal::Int, prob::Real, eta::Real, b::Int=lsize ÷ 2, which_ent::Real=1; 
    numsamp::Int=10, cutoff::Real=1e-14, ent_cutoff::Real=1e-12)
    """
    Calculate the mean entanglement entropy over multiple samples.
    """
    ss = siteinds("S=1/2", lsize)
    psi0 = MPS(ss, "Up")
    entropies = Float64[]
    for _ in 1:numsamp 
        psi = mps_evolve(psi0, ttotal, prob, eta; cutoff=cutoff)
        entropy = Renyi_entropy(psi, b, which_ent; cutoff=ent_cutoff)
        push!(entropies, entropy)
    end
    return sum(entropies) / numsamp
end

function entropy_mean(lsize::Int, ttotal::Int, prob::Real, para::Tuple{Real, Real}, b::Int=lsize ÷ 2, which_ent::Real=1; 
    numsamp::Int=10, cutoff::Real=1e-14, ent_cutoff::Real=1e-12)
    """
    Calculate the mean entanglement entropy over multiple samples. (weak measurement case)
    """
    ss = siteinds("S=1/2", lsize)
    psi0 = MPS(ss, "Up")
    entropies = Float64[]
    for _ in 1:numsamp 
        psi = mps_evolve(psi0, ttotal, prob, para; cutoff=cutoff)
        entropy = Renyi_entropy(psi, b, which_ent; cutoff=ent_cutoff)
        push!(entropies, entropy)
    end
    return sum(entropies) / numsamp
end

let
    Ls = 10:10:50
    p0, η0 = 0.5, 0.5
    ps = 0.0:0.05:1.0
    ηs = 0.0:0.5:2.0

    prob_scales = []
    eta_scales = []

    for l in Ls
        tt = 4l
        b = l ÷ 2
        entropy_prob = [entropy_mean(l, tt, p, η0, b; numsamp=10) for p in ps]
        push!(prob_scales, entropy_prob)

        entropy_eta = [entropy_mean(l, tt, p0, η, b; numsamp=10) for η in ηs]
        push!(eta_scales, entropy_eta)
    end

    prob_scales = hcat(prob_scales...)
    eta_scales = hcat(eta_scales...)

    h5open("entropy_scale_data.h5", "w") do file
        write(file, "ps", collect(ps))
        write(file, "ηs", collect(ηs))
        write(file, "Ls", collect(Ls))
        write(file, "prob_scales", prob_scales)
        write(file, "eta_scales", eta_scales)
    end
end