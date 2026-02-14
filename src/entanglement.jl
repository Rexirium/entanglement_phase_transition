using NDTensors

function ent_entropy(ps::NDTensors.Tensor, n::Real=1)::Real
    """
    Calculate the n-th order Renyi entropy from the eigenvalues ps.
    """
    if n == 0
        return log2(length(ps))
    elseif n == 1
        return -sum(ps .* log2.(ps))
    else
        trace = sum(ps .^ n)
        return log2(trace)/(1-n)
    end
end

function schmidt_decomp(psi::MPS, b::Int)::NDTensors.Tensor
    """
    Perform Schmidt decomposition of the MPS `psi` biparted after site `b`.
    Returns the Schmidt coefficients as a vector.
    """
    # SVD decomposition to obtain the Schmidt coefficients
    orthogonalize!(psi, b)
    linds = uniqueinds(psi[b], psi[b+1])
    _ , S, _ = ITensors.svd(psi[b], linds)
    schs = diag(S)

    return schs[schs .> 0.0] # remove zero probabilities
end

function ent_entropy(psi::MPS, b::Int, n::Real=1)
    """
    Calculate the n-th order Renyi entropy of the MPS `psi` biparted after site `b`.
    """
    (b <= 0 || b >= length(psi)) && return 0.0
    # SVD decomposition to obtain the Schmidt coefficients
    schs = schmidt_decomp(psi, b)
    return ent_entropy(schs .* schs, n)
end

function ent_entropy(psi::MPS, xs::Vector{<:Int}, n::Real=1)
    """
    Calculate the n-th order Renyi entropy of a region of sites `xs` from other sites.
    """
    sort!(xs)
    (xs[1] ≤ 0 || xs[end] > length(psi)) && error("The sites do not exist!")
    ps = reduced_density_eigen(psi, xs)
    return ent_entropy(ps, n)
end

function negativity(psi::MPS, b::Int; logscale::Bool=true)
    """
    Calculate the entanglement negativity of the MPS `psi` biparted after site `b`
    """
    (b <= 0 || b >= length(psi)) && return 0.0
    schs = schmidt_decomp(psi, b)
    modulus = sum(schs)
    if logscale
        return 2 * log2(modulus)
    else
        return (modulus * modulus - 1)/2
    end
end

function negativity(psi::MPS, xs::Vector{<:Int}; logscale::Bool=true)
    """
    Calculate the entanglement negativity of a region of sites `xs` from other sites.
    """
    sort!(xs)
    (xs[1] ≤ 0 || xs[end] > length(psi)) && error("The sites do not exist!")

    ps = reduced_density_eigen(psi, xs)
    modulus = sum(sqrt.(ps))
    if logscale
        return 2 * log2(modulus)
    else
        return (modulus * modulus - 1)/2
    end
end

function concurrence(psi::MPS, b::Int)
    (b <= 0 || b >= length(psi)) && return 0.0
    schs = schmidt_decomp(psi, b)
    trace = sum(schs .^ 4)
    return sqrt(2 * abs(1 - trace))
end

function concurrence(psi::MPS, xs::Vector{<:Int})
    sort!(xs)
    (xs[1] ≤ 0 || xs[end] > length(psi)) && error("The sites do not exist!")

    ps = reduced_density_eigen(psi, xs)
    trace = sum(ps .^ 2)
    return sqrt(2 * abs(1 - trace))
end

function reduced_density_eigen(psi::MPS, x::Int)::NDTensors.Tensor
    """
    Calculate the reduced density matrix eigen values of a region of a single sites `x` from other sites.
    """
    # obtain the reduced density matrix 
    orthogonalize!(psi, x)
    Ap = prime(dag(psi[x]), tags="Site")
    rho = contract(Ap, psi[x])
    # diagonalize the reduced density matrix
    D, _ = eigen(rho; ishermitian=true)
    ps = diag(D)
    return ps[ps .> 0.0]  # remove zero probabilities
end

function reduced_density_eigen(psi::MPS, xs::Vector{<:Int})::NDTensors.Tensor
    """
    Calculate the reduced density matrix eigen values of multiple sites `xs` from other sites.
    """
    length(xs) == 1 && return reduced_density_eigen(psi, xs[1])
    a, b = xs[1], xs[end]
    # obtain the reduced density matrix
    orthogonalize!(psi, a)
    rho = psi[a]
    ir = linkind(psi, a)
    rho *= dag(prime(prime(psi[a], tags="Site"), ir))
    for j in (a+1):(b-1)
        rho *= psi[j]
        if j in xs
            rho *= dag(prime(psi[j]))
        else
            rho *= dag(prime(psi[j], tags="Link"))
        end
    end
    rho *= psi[b]
    il = linkind(psi, b-1)
    rho *= dag(prime(prime(psi[b], tags="Site"), il))
    # diagonalize the reduced density matrix
    D, _ = eigen(rho; ishermitian=true)
    ps = diag(D)
    return ps[ps .> 0.0]  # remove zero probabilities
end

function von_neumann_entropy(psi::MPS, b::Int)
    """
    Calculate the von Neumann entropy of the MPS `psi` biparted after site `b`.
    """
    return ent_entropy(psi, b, 1)
end

function zeroth_entropy(psi::MPS, b::Int)
    """
    Calculate the zeroth order Renyi entropy of the MPS `psi` biparted after site `b`.
    """
    (b <= 0 || b >= length(psi)) && return 0.0
    orthogonalize!(psi, b)
    truncate!(psi, b) 
    return log2(linkdim(psi, b))
end

function renyi_entropy(psi::MPS, b::Int, n::Real)
    """
    Calculate the n-th order Renyi entropy of the MPS `psi` biparted after site `b`.
    """
    return ent_entropy(psi, b, n)
end

function von_neumann_entropy(psi::MPS, xs::Vector{<:Int})
    """
    Calculate the von Neumann entropy of a region of sites `xs` from other sites.
    """
    return ent_entropy(psi, xs, 1)
end

function zeroth_entropy(psi::MPS, xs::Vector{<:Int})
    """
    Calculate the zeroth order Renyi entropy of a region of sites `xs` from other sites.
    """
    return ent_entropy(psi, xs, 0)
end

function renyi_entropy(psi::MPS, xs::Vector{<:Int}, n::Real)
    """
    Calculate the n-th order Renyi entropy of a region of sites `xs` from other sites.
    """
    return ent_entropy(psi, xs, n)
end

function mutual_information(psi::MPS, as::Vector{<:Int}, bs::Vector{<:Int}, n::Real=1)
    """
    Calculate the mutual information of two separate region of sites `as` and `bs`.
    """
    Sa = ent_entropy(psi, as, n)
    Sb = ent_entropy(psi, bs, n)
    Sab = ent_entropy(psi, sort(union(as, bs)), n)
    return Sa + Sb - Sab
end
