function ITensors.op(::OpName"RdU", ::SiteType"S=1/2", s::Index...; eltype::DataType=ComplexF64)
    """
    Create a random unitary operator for the given site indices `s`.
    """
    M = randn(eltype, 4, 4)
    Q, _ = NDTensors.qr_positive(M)
    return op(Q, s...)
end

function ITensors.op(::OpName"NH", ::SiteType"S=1/2", s::Index; eta::Real)
    """
    Create a non-Hermitian operator for the given site index `s` with parameter `eta`.
    """
    M = diagm(shuffle([one(eta), eta]))
    return op(M, s)
end

function ITensors.op(::OpName"WM", ::SiteType"S=1/2", s::Index; x::Real, λ::Real=1.0, Δ::Real=1.0)
    """Create a weak measurement operator for the given site index `s` with parameters `x`, `λ`, and `Δ`."""
    # Assuming `x` is a random variable from a Gaussian distribution
    phiUp = exp(-((x - λ) / (2*Δ)) ^ 2)
    phiDn = exp(-((x + λ) / (2*Δ)) ^ 2)
    return op([phiUp 0; 0 phiDn], s)
end

function proj_measure!(psi::MPS, loc::Int)
    (loc <= 0 || loc > length(psi)) && return psi
    # Orthogonalize the MPS at site `loc`
    orthogonalize!(psi, loc)
    # Calculate the probability of measuring "Up"
    s = siteind(psi, loc)
    proj = ITensor(s', s)
    proj[s'=>1, s=>1] = 1.0

    Aup = noprime(psi[loc] * proj)
    probUp = real(scalar(Aup * dag(psi[loc])))
    # Collapse the states
    if rand() < probUp
        psi[loc] = Aup
        normalize!(psi)
        return true
    else
        proj[s'=>1, s=>1] = 0.0
        proj[s'=>2, s=>2] = 1.0

        psi[loc] *= proj
        noprime!(psi[loc])
        normalize!(psi)
        return false
    end
end

function proj_measure!(psi::MPS, phi::MPS, loc::Int)
    (loc <= 0 || loc > length(psi)) && return psi
    # Orthogonalize the MPS at site `loc`
    orthogonalize!(psi, loc)
    orthogonalize!(phi, loc)
    # Calculate the probability of measuring "Up"
    s = siteind(psi, loc)
    proj = ITensor(s', s)
    proj[s'=>1, s=>1] = 1.0

    Aup = noprime(psi[loc] * proj)
    probUp = real(scalar(Aup * dag(psi[loc])))
    # Collapse the states
    if rand() < probUp
        psi[loc] = Aup
        normalize!(psi)
        phi[loc] *= proj
        noprime!(phi[loc])
        normalize!(phi)
        return true
    else
        proj[s'=>1, s=>1] = 0.0
        proj[s'=>2, s=>2] = 1.0

        psi[loc] *= proj
        noprime!(psi[loc])
        normalize!(psi)
        phi[loc] *= proj
        noprime!(phi[loc])
        normalize!(phi)
        return false
    end
end

function weak_measure!(psi::MPS, loc::Int, para::Tuple{T, T}=(1.0, 1.0)) where T<:Real
    """Perform a weak measurement on the MPS `psi` at site `loc` with parameters `λ` and `Δ`."""
    (loc <= 0 || loc > length(psi)) && return psi
    # Orthogonalize the MPS at site `loc`
    orthogonalize!(psi, loc)
    s = siteind(psi, loc)
    λ, Δ = para
    
    proj = ITensor(s)
    proj[s => 1] = 1.0
    Aup = psi[loc] * dag(proj)
    # Calculate the probability of measuring "Up"
    probUp = real(scalar(dag(Aup) * Aup))
    # generate a random variable from a Gaussian distribution
    x = rand(T) < probUp ? λ + Δ*randn(T) : -λ + Δ*randn(T)
    M = op("WM", s; x = x, λ = λ, Δ = Δ)
    # Apply the weak measurement operator
    psi[loc] *= M
    noprime!(psi[loc])
    normalize!(psi)
    return x
end

function applyn!(G::ITensor, psi::MPS; cutoff::Real=1e-14, maxdim::Int=4*maxlinkdim(psi))
    js = findsites(psi, G)
    return applyn!(G, psi, js...; cutoff=cutoff, maxdim=maxdim)
end

function applyn!(G1::ITensor, psi::MPS, loc::Int; cutoff::Real=1e-14, maxdim::Int=4*maxlinkdim(psi))
    """
    Apply the gate `G1` to the MPS `psi` at site `loc` inplace.
    """
    orthogonalize!(psi, loc)
    psi[loc] *= G1
    noprime!(psi[loc])
    return 0.0
end

function applyn!(G2::ITensor, psi::MPS, j1::Int, j2::Int; cutoff::Real=1e-14, maxdim::Int=4*maxlinkdim(psi))
    """
    Apply two adjacent site gate `G2` to the MPS `psi` at sites `j1` and `j1+1` inplace.
    """
    (j1<=0 || j1>= length(psi)) && error("Wrong starting site for two-site gate application.")
    orthogonalize!(psi, j1)

    A = (psi[j1] * psi[j2]) * G2
    noprime!(A)
    linds = uniqueinds(psi[j1], psi[j2])
    psi[j1], S, psi[j2], spec = svd(A, linds; cutoff=cutoff, maxdim=maxdim)
    psi[j2] *= S

    replacetags!(psi[j1], "Link,u" => "Link,l=$j1")
    replacetags!(psi[j2], "Link,u" => "Link,l=$j1")
    set_ortho_lims!(psi, j2:j2)

    return spec.truncerr
end

function applyn!(G3::ITensor, psi::MPS, j1::Int, j2::Int, j3::Int; cutoff::Real=1e-14, maxdim::Int=4*maxlinkdim(psi))
    """
    Apply three adjacent site gate `G3` to the MPS `psi` at sites `j2-1`, `j2`, and `j2+1` inplace.
    """
    (j2 <= 1 || j2 >= length(psi)) && error("Wrong middle site for three-site gate application.")
    orthogonalize!(psi, j1)
    s = siteind(psi, j2)
    
    A = (psi[j1] * psi[j2] * psi[j3]) * G3
    noprime!(A)
    linds12 = uniqueinds(psi[j1], psi[j2])
    psi[j1], S12, B, spec12 = svd(A, linds12; cutoff=cutoff, maxdim=maxdim)
    B *= S12
    replacetags!(psi[j1], "Link,u" => "Link,l=$j1")
    replacetags!(B, "Link,u" => "Link,l=$j1")

    linds23 = (commonind(psi[j1], B), s)
    psi[j2], S23, psi[j3], spec23 = svd(B, linds23; cutoff=cutoff, maxdim=maxdim)
    psi[j3] *= S23
    
    replacetags!(psi[j2], "Link,u" => "Link,l=$j2")
    replacetags!(psi[j3], "Link,u" => "Link,l=$j2")
    set_ortho_lims!(psi, j3:j3)

    return spec12.truncerr + spec23.truncerr
end

function applyn!(Gs::Vector{ITensor}, psi::MPS; cutoff::Real=1e-14, maxdim::Int=4*maxlinkdim(psi))
    """
    Apply a vector of gates `Gs` to the MPS `psi` inplace.
    """
    truncerr = 0.0
    for G in Gs
        truncerr += applyn!(G, psi; cutoff=cutoff, maxdim=maxdim)
    end
    return truncerr
end

