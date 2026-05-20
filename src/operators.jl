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

function applyn!(G::ITensor, psi::MPS; cutoff::Real=1e-14, maxdim::Int=4*maxlinkdim(psi), rev::Bool=false)
    js = findsites(psi, G)
    return applyn!(G, psi, js...; cutoff=cutoff, maxdim=maxdim, rev=rev)
end

function applyn!(G1::ITensor, psi::MPS, j::Int; cutoff::Real=1e-14, maxdim::Int=4*maxlinkdim(psi), rev::Bool=false)
    """
    Apply the gate `G1` to the MPS `psi` at site `j` inplace.
    """
    orthogonalize!(psi, j)
    psi[j] *= G1
    noprime!(psi[j])
    return 0.0
end

function applyn!(G2::ITensor, psi::MPS, j1::Int, j2::Int; cutoff::Real=1e-14, maxdim::Int=4*maxlinkdim(psi), rev::Bool=false)
    """
    Apply two adjacent site gate `G2` to the MPS `psi` at sites `j1` and `j1+1` inplace.
    """
    (j1<=0 || j1>= length(psi)) && error("Wrong starting site for two-site gate application.")
    ja, jb = rev ? (j2, j1) : (j1, j2)
    orthogonalize!(psi, ja)

    A = (psi[ja] * psi[jb]) * G2
    noprime!(A)
    indsab = uniqueinds(psi[ja], psi[jb])
    psi[ja], S, psi[jb], spec = svd(A, indsab; cutoff=cutoff, maxdim=maxdim)
    psi[jb] *= S

    replacetags!(psi[ja], "Link,u" => "Link,l=$j1")
    replacetags!(psi[jb], "Link,u" => "Link,l=$j1")
    set_ortho_lims!(psi, jb:jb)

    return spec.truncerr
end

function applyn!(G3::ITensor, psi::MPS, j1::Int, j2::Int, j3::Int; cutoff::Real=1e-14, maxdim::Int=4*maxlinkdim(psi), rev::Bool=false)
    """
    Apply three adjacent site gate `G3` to the MPS `psi` at sites `j2-1`, `j2`, and `j2+1` inplace.
    """
    (j2 <= 1 || j2 >= length(psi)) && error("Wrong middle site for three-site gate application.")
    ja, jb, jc = rev ? (j3, j2, j1) : (j1, j2, j3)
    jab, jbc = rev ? (j2, j1) : (j1, j2)
    orthogonalize!(psi, ja)
    s = siteind(psi, jb)

    A = (psi[ja] * psi[jb] * psi[jc]) * G3
    noprime!(A)

    indsab = uniqueinds(psi[ja], psi[jb])
    psi[ja], Sab, B, specab = svd(A, indsab; cutoff=cutoff, maxdim=maxdim)
    B *= Sab
    
    replacetags!(psi[ja], "Link,u" => "Link,l=$jab")
    replacetags!(B, "Link,u" => "Link,l=$jab")

    indsbc = (commonind(psi[ja], B), s)
    psi[jb], Sbc, psi[jc], specbc = svd(B, indsbc; cutoff=cutoff, maxdim=maxdim)
    psi[jc] *= Sbc
    
    replacetags!(psi[jb], "Link,u" => "Link,l=$jbc")
    replacetags!(psi[jc], "Link,u" => "Link,l=$jbc")
    set_ortho_lims!(psi, jc:jc)

    return specab.truncerr + specbc.truncerr
end

function applyn!(Gs::Vector{ITensor}, psi::MPS; cutoff::Real=1e-14, maxdim::Int=4*maxlinkdim(psi), rev::Bool=false)
    """
    Apply a vector of gates `Gs` to the MPS `psi` inplace.
    """
    truncerr = 0.0
    for G in Gs
        truncerr += applyn!(G, psi; cutoff=cutoff, maxdim=maxdim, rev=rev)
    end
    return truncerr
end

