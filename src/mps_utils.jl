#================== Infinite MPS ======================#

struct InfMPS
    len_uc::Int
    Gammas::Vector{ITensor}
    Lambdas::Vector{ITensor}
end

function InfMPS(ss::Vector{<:Index}, state::Function)
    len_uc = length(ss)

    ls = [Index(1, "LLink,l=$n") for n in 1:len_uc]
    rs = [Index(1, "RLink,r=$n") for n in 1:len_uc]

    Γs = ITensor[]
    Λs = ITensor[]
    for n in 1 : len_uc
        sn = state(n) == "Up" ? 1 : 2
        nn = mod1(n+1, len_uc)

        Γ = ITensor(ls[n], ss[n], rs[n])
        Λ = ITensor(rs[n], ls[nn])

        Γ[ls[n]=>1, ss[n]=>sn, rs[n]=>1] = 1.0
        Λ[rs[n]=>1, ls[nn]=>1] = 1.0

        push!(Γs, Γ)
        push!(Λs, Λ)
    end
    return InfMPS(len_uc, Γs, Λs)
end

function InfMPS(ss::Vector{<:Index}, state::String)
    statefunc(n::Int) = state
    return InfMPS(ss, statefunc)
end

function ITensorMPS.siteinds(psi::InfMPS)
    ss = Index[]
    for n in 1:(psi.len_uc)
        push!(ss, first(inds(psi.Gammas[n], "Site")))
    end
    return ss
end

function ITensorMPS.linkinds(psi::InfMPS)
    lkinds = Index[]
    for n in 1:(psi.len_uc)
        push!(lkinds, first(inds(psi.Gammas[n], "LLink")))
        push!(lkinds, first(inds(psi.Gammas[n], "RLink")))
    end
    return lkinds
end

function ITensorMPS.maxlinkdim(psi::InfMPS)
    lkinds = linkinds(psi)
    return maximum(dim.(lkinds))
end

function leftlinkinds(psi::InfMPS)
    ls = Index[]
    for n in 1:(psi.len_uc)
        push!(ls, first(inds(psi.Gammas[n], "LLink")))
    end
    return ls
end

function rightlinkinds(psi::InfMPS)
    rs = Index[]
    for n in 1:(psi.len_uc)
        push!(rs, first(inds(psi.Gammas[n], "RLink")))
    end
    return rs
end

function findsites(psi::InfMPS, o::ITensor)
    ss = siteinds(psi)
    so = inds(o)
    return findall(ss, so)
end

#================ Apply functions ===================#

function applyn!(G::ITensor, psi::Union{MPS, InfMPS}; cutoff::Real=1e-14, maxdim::Int=typemax(Int), rev::Bool=false)
    js = findsites(psi, G)
    return applyn!(G, psi, js...; cutoff=cutoff, maxdim=maxdim, rev=rev)
end

function applyn!(Gs::Vector{ITensor}, psi::Union{MPS, InfMPS}; cutoff::Real=1e-14, 
    maxdim::Int=typemax(Int), rev::Bool=false)
    """
    Apply a vector of gates `Gs` to the MPS `psi` inplace.
    """
    truncerr = 0.0
    for G in Gs
        truncerr += applyn!(G, psi; cutoff=cutoff, maxdim=maxdim, rev=rev)
    end
    return truncerr
end

# Finite MPS version
function applyn!(G1::ITensor, psi::MPS, j::Int; cutoff::Real=1e-14, maxdim::Int=typemax(Int), rev::Bool=false)
    """
    Apply the gate `G1` to the MPS `psi` at site `j` inplace.
    """
    orthogonalize!(psi, j)
    psi[j] *= G1
    noprime!(psi[j], "Site")
    return 0.0
end

function applyn!(G2::ITensor, psi::MPS, j1::Int, j2::Int; cutoff::Real=1e-14, maxdim::Int=typemax(Int), rev::Bool=false)
    """
    Apply two adjacent site gate `G2` to the MPS `psi` at sites `j1` and `j1+1` inplace.
    """
    (j1 < 1 || j2 > length(psi)) && error("Wrong starting site for two-site gate application.")
    maximum(diff([j1, j2])) > 1 && error("Only adjacent sites are allowed for applyn!")

    ja, jb = rev ? (j2, j1) : (j1, j2)
    orthogonalize!(psi, ja)

    A = (psi[ja] * psi[jb]) * G2
    noprime!(A, "Site")
    indsab = uniqueinds(psi[ja], psi[jb])
    psi[ja], S, psi[jb], spec = svd(A, indsab; cutoff=cutoff, maxdim=maxdim)
    psi[jb] *= S

    replacetags!(psi[ja], "Link,u" => "Link,l=$j1")
    replacetags!(psi[jb], "Link,u" => "Link,l=$j1")
    set_ortho_lims!(psi, jb:jb)

    return spec.truncerr
end

function applyn!(G3::ITensor, psi::MPS, j1::Int, j2::Int, j3::Int; cutoff::Real=1e-14, 
    maxdim::Int=typemax(Int), rev::Bool=false)
    """
    Apply three adjacent site gate `G3` to the MPS `psi` at sites `j2-1`, `j2`, and `j2+1` inplace.
    """
    (j1 < 1 || j3 > length(psi)) && error("Wrong middle site for three-site gate application.")
    maximum(diff([j1, j2, j3])) > 1 && error("Only adjacent sites are allowed for applyn!")

    ja, jb, jc = rev ? (j3, j2, j1) : (j1, j2, j3)
    jab, jbc = rev ? (j2, j1) : (j1, j2)
    orthogonalize!(psi, ja)
    s = siteind(psi, jb)

    A = (psi[ja] * psi[jb] * psi[jc]) * G3
    noprime!(A, "Site")
    
    truncerr = 0.0
    indsab = uniqueinds(psi[ja], psi[jb])
    psi[ja], Sab, B, spec = svd(A, indsab; cutoff=cutoff, maxdim=maxdim)
    truncerr += spec.truncerr
    B *= Sab
    
    replacetags!(psi[ja], "Link,u" => "Link,l=$jab")
    replacetags!(B, "Link,u" => "Link,l=$jab")

    indsbc = (commonind(psi[ja], B), s)
    psi[jb], Sbc, psi[jc], spec = svd(B, indsbc; cutoff=cutoff, maxdim=maxdim)
    truncerr += spec.truncerr
    psi[jc] *= Sbc
    
    replacetags!(psi[jb], "Link,u" => "Link,l=$jbc")
    replacetags!(psi[jc], "Link,u" => "Link,l=$jbc")
    set_ortho_lims!(psi, jc:jc)

    return truncerr
end

# Infinite MPS version
function inv_tensor(A::ITensor)
    r, l = inds(A)
    invA = ITensor(l, r)
    mindim = min(dim(r), dim(l))
    for i in 1 : mindim
        val = A[r=>i, l=>i]
        invval = val > 1e-300 ? 1.0 / val : 1e300
        invA[l=>i, r=>i] = invval
    end
    return invA
end

function applyn!(G::ITensor, psi::InfMPS, j::Int; cutoff::Real=1e-14, maxdim::Int=typemax(Int))
    psi.Gammas[j] *= G
    noprime!(psi.Gammas[j], "Site")
    return 0.0
end

function applyn!(G::ITensor, psi::InfMPS, j1::Int, j2::Int; cutoff::Real=1e-14, 
    maxdim::Int=typemax(Int), rev::Bool=false)
    """
    Apply two adjacent site gate `G2` to the InfMPS `psi` at sites `j1` and `j1+1` inplace.
    """
    j0 = mod1(j1 - 1, psi.len_uc)

    Λ0 = prime(psi.Lambdas[j0], "RLink")
    Λ1 = psi.Lambdas[j1]
    Λ2 = prime(psi.Lambdas[j2], "LLink")
    Γs = psi.Gammas[j1 : j2]
    
    Θ = Λ0 * Γs[1] * Λ1 * Γs[2] * Λ2
    Θ *= G
    noprime!(Θ, "Site")
    invΛ0 = inv_tensor(Λ0)
    invΛ2 = inv_tensor(Λ2)

    inds12 = (ind(Λ0, 1), first(inds(Γs[1], "Site")))
    U, S, V, spec = svd(Θ, inds12; cutoff=cutoff, maxdim=maxdim)
    U *= invΛ0
    V *= invΛ2

    replacetags!(S, "Link,u"=>"RLink,r=$j1")
    replacetags!(S, "Link,v"=>"LLink,l=$j2")
    psi.Lambdas[j1] = S

    psi.Gammas[j1] = replacetags(U, "Link,u"=>"RLink,r=$j1")
    psi.Gammas[j2] = replacetags(V, "Link,v"=>"LLink,l=$j2")

    return spec.truncerr
end

function applyn!(G::ITensor, psi::InfMPS, j1::Int, j2::Int, j3::Int; cutoff::Real=1e-14, 
    maxdim::Int=typemax(Int), rev::Bool=false)
    """
    Apply three adjacent site gate `G2` to the InfMPS `psi` at sites `j1`, `j2` and `j3` inplace.
    """

    j0 = mod1(j1 - 1, psi.len_uc)

    Λ0 = prime(psi.Lambdas[j0], "RLink")
    Λ3 = prime(psi.Lambdas[j3], "LLink")
    Λs = psi.Lambdas[j1 : j2]
    Γs = psi.Gammas[j1 : j3]

    Θ = Λ0
    for j in 1 : 2
        Θ *= Γs[j] * Λs[j]
    end
    Θ *= Γs[3] * Λ3
    Θ *= G
    noprime!(Θ, "Site")
    invΛ0 = inv_tensor(Λ0)
    invΛ3 = inv_tensor(Λ3)

    truncerr = 0.0
    inds12 = (ind(Λ0, 1), first(inds(Γs[1], "Site")))
    U, S12, B, spec = svd(Θ, inds12; cutoff=cutoff, maxdim=maxdim)
    truncerr += spec.truncerr
    U *= invΛ0

    replacetags!(S12, "Link,u"=>"RLink,r=$j1")
    replacetags!(S12, "Link,v"=>"LLink,l=$j2")
    replacetags!(B, "Link,v"=>"LLink,l=$j2")

    psi.Gammas[j1] = replacetags(U, "Link,u"=>"RLink,r=$j1")
    psi.Lambdas[j1] = S12

    inds23 = (commonind(S12, B), first(inds(Γs[2], "Site")))
    U, S23, V, spec = svd(B, inds23; cutoff=cutoff, maxdim=maxdim)
    truncerr += spec.truncerr
    V *= invΛ3

    replacetags!(S23, "Link,u"=>"RLink,r=$j2")
    replacetags!(S23, "Link,v"=>"LLink,l=$j3")

    psi.Lambdas[j2] = S23
    psi.Gammas[j2] = replacetags(U, "Link,u"=>"RLink,r=$j2")
    psi.Gammas[j3] = replacetags(V, "Link,v"=>"LLink,l=$j3")

    return truncerr
    
end