using LinearAlgebra
import Base: length


pown1(x::Integer) = isodd(x) ? -1 : 1


###############################  FiniteArray  #################################

@enum PaddingType  Zero Cycle Linear SymmetryInt SymmetryHalf

mutable struct FiniteArray{T<:Number} <: AbstractVector{T}
    idxstart::Int
    values::Vector{T}
    padding::PaddingType
end

firstindex(fa::FiniteArray) = fa.idxstart
lastindex(fa::FiniteArray) = fa.idxstart + length(fa.values) - 1
eachindex(fa::FiniteArray) = firstindex(fa):lastindex(fa)
eltype(fa::FiniteArray{T}) where T = T

Base.size(fa::FiniteArray) = size(fa.values)
Base.length(fa::FiniteArray) = length(fa.values)
Base.setindex!(fa::FiniteArray, v::Number, i) = fa.values[i - fa.idxstart + 1] = v

function Base.getindex(fa::FiniteArray{T}, i) where T
    idx = i - fa.idxstart + 1
    len = length(fa)
    (1 ≤ idx ≤ len) && return fa.values[idx]
    (fa.padding == Zero) && return zero(T)
    (fa.padding == Cycle) && return fa.values[mod(idx, 1:len)]
    if fa.padding == Linear
        (len == 1) && return fa.values[1]
        (idx < 1) && return fa.values[1] * (2 - idx) - fa.values[2] * (1 - idx)
        return fa.values[len] * (1 + idx - len) - fa.values[len - 1] * (idx - len)
    elseif fa.padding == SymmetryInt || fa.padding == SymmetryHalf
        tmp = (fa.padding == SymmetryInt) ? 1 : 0
        n, shift = divrem((idx > 0) ? (idx - 1) : (tmp - idx), len - tmp)
        return iseven(n) ? fa.values[1 + shift] : fa.values[len - shift]
    end
end
Base.getindex(fa::FiniteArray, idces::AbstractArray) = map(idx -> fa[idx], idces)

FiniteArray{T}(init::UndefInitializer, idxstart, idxend; padding = Zero) where T <: Number =
        FiniteArray(idxstart, Vector{T}(init, idxend - idxstart + 1), padding)


###########################  Scale coefficients {pₖ}  #########################

CoefP(coef...; idxstart::Integer = 0) = FiniteArray(idxstart, collect(coef), Zero)


#################  First type of function approximation  ######################

CoefC(n::Integer) = (n ≤ 0) ? (FiniteArray(0, [1], Zero), 0) :
        throw(ArgumentError("`p` must be given when `n` > 0."))
function CoefC(n::Integer, p::FiniteArray)
    result = FiniteArray(0, [1], Zero)
    for nn ∈ 1:n
        result, _ = nextCoefC(result, nn - 1, p)
    end
    return result, n
end

function nextCoefC(c::FiniteArray{U}, n::Integer, p::FiniteArray{V}) where {U, V}
    cf = firstindex(c); cl = lastindex(c)
    pf = firstindex(p); pl = lastindex(p)
    tmp = 2^n
    outf = tmp * pf + cf; outl = tmp * pl + cl
    result = FiniteArray{promote_type(U, V)}(undef, outf, outl)
    for l ∈ outf:outl
        kf = max(pf, ceil(Int, (l - cl) / tmp))
        kl = min(pl, floor(Int, (l - cf) / tmp))
        result[l] = sum(p[k] * c[l - tmp * k] for k ∈ kf:kl)
    end
    return result, n + 1
end

function callCoefC(c::FiniteArray{T}, n::Integer) where T
    tmp = 2^n
    return x::Real -> c[floor(Int, tmp * x)]
end


#################  Second type of function approximation  #####################

function BinPoint(n::Integer, p::FiniteArray)
    (n < 0) && throw(ArgumentError("only support `n` >= 0"))
    result, _ = IntPointValues(p)
    for nn ∈ 1:n
        result, _ = nextBinPointValues(result, nn - 1, p)
    end
    return result, n
end

function IntPoint(p::FiniteArray)
    len = length(p)
    P = [p[2a - b] for a ∈ 1:len - 2, b ∈ 1:len - 2]
    b = nullspace(P - I; rtol = 1e-4)[:, 1]
    return FiniteArray(1, b ./ sum(b), Zero), 0
end

function nextBinPoint(b::FiniteArray{U}, n::Integer, p::FiniteArray{V}) where {U, V}
    bf = firstindex(b); bl = lastindex(b)
    pf = firstindex(p); pl = lastindex(p)
    tmp = 2^n
    result = Vector{promote_type(U, V)}(undef, 2length(v) + 1)
    result[2:2:end] = b.values
    for l ∈ 1:2:length(result)
        kf = max(pf, ceil(Int, (l - bl) / tmp))
        kl = min(pl, floor(Int, (l - bf) / tmp))
        result[l] = sum(p[k] * b[l - tmp * k] for k ∈ kf:kl)
    end
    return FiniteArray(1, result, Zero), n + 1
end

function callBinPoint(b::FiniteArray, n::Integer)
    tmp = 2^n
    return x::Real -> begin
        tmp1 = tmp * x
        idx = floor(Int, tmp1)
        subx = tmp1 - idx
        return b[idx] * (1 - subx) + b[idx + 1] * subx
    end
end


############################  Wavelet function   ##############################

function getwavelet(v::FiniteArray{U}, n::Integer, p::FiniteArray{V}) where {U, V}
    vf = firstindex(v); vl = lastindex(v)
    pf = firstindex(p); pl = lastindex(p)
    tmp = 2^n
    outf = vf - tmp * (pl - 1); outl = vl - tmp * (pf - 1)
    result = FiniteArray{promote_type(U, V)}(undef, outf, outl)
    for l ∈ outf:outl
        kf = max(pf, ceil(Int, (vf - l) / tmp) + 1)
        kl = min(pl, floor(Int, (vl - l) / tmp) + 1)
        result[l] = sum(pown1(k - 1) * conj(p[k]) * v[l + tmp * (k - 1)] for k ∈ kf:kl)
    end
    return result, n + 1
end



################################  Showing  ####################################
using Plots

Daubechies = (
    CoefP(1., 1.),
    CoefP((1+√3)/4, (3+√3)/4, (3-√3)/4, (1-√3)/4),
    CoefP(0.47046721, 1.14111692, 0.650365, -0.19093442, -0.12083221, 0.0498175),
    CoefP(0.32580343, 1.01094572, 0.89220014, -0.03957503, -0.26450717, 0.0436163,
            0.0465036, -0.01498699),
    CoefP(0.22641898, 0.85394354, 1.02432694, 0.19576696, -0.34265671, -0.04560113,
            0.10970265, -0.00882680, -0.01779187, 4.71742793e-3),
    CoefP(0.15774243, 0.69950381, 1.06226376, 0.44583132, -0.31998660, -0.18351806,
            0.13788809, 0.03892321, -0.04466375, 7.83251152e-4, 6.75606236e-3,
            -1.52353381e-3),
    CoefP(0.11009943, 0.56079128, 1.03114849, 0.66437248, -0.20351382, -0.31683501,
            0.1008467, 0.11400345, -0.05378245, -0.02343994, 0.01774979, 6.07514995e-4,
            -2.54790472e-3, 5.00226853e-4),
    CoefP(0.07695562, 0.44246725, 0.95548615, 0.82781653, -0.02238574, -0.40165863,
            6.68194092e-4, 0.18207636, -0.02456390, -0.06235021, 0.01977216, 0.01236884,
            -6.88771926e-3, -5.54004549e-4, 9.55229711e-4, -1.66137261e-4),
    CoefP(0.05385035, 0.34483430, 0.85534906, 0.92954571, 0.18836955, -0.41475176,
            -0.13695355, 0.21006834, 0.043452675, -0.09564726, 3.54892813e-4, 0.03162417,
            -6.67962023e-3, -6.05496058e-3, 2.61296728e-3, 3.25814671e-4, -3.56329759e-4,
            5.5645514e-5),
    CoefP(0.03771716, 0.26612218, 0.74557507, 0.97362811, 0.39763774, -0.35333620,
            -0.27710988, 0.18012745, 0.13160299, -0.10096657, -0.04165925, 0.04696981,
            5.10043697e-3, -0.01517900, 1.97332536e-3, 2.81768659e-3, -9.69947840e-4,
            -1.64709006e-4, 1.32354367e-4, -1.875841e-5)
)

N = 2
n = 4

x = -N:0.001:2N

c, _ = CoefC(0)
plot(callCoefC(c, 0), x)
for nn ∈ 1:n
    global c, _ = nextCoefC(c, nn - 1, Daubechies[N])
    plot!(callCoefC(c, nn), x)
end
display(title!("function approximation 1"))

v, _ = IntPoint(Daubechies[N])
fig = plot(callBinPoint(v, 0), x)
for nn ∈ 1:n
    global v, _ = nextBinPoint(v, nn - 1, Daubechies[N])
    plot!(callBinPoint(v, nn), x)
end
display(title!("function approximation 2"))

plot(callCoefC(getwavelet(c, n, Daubechies[N])...), x)
plot!(callBinPoint(getwavelet(v, n, Daubechies[N])...), x)
display(title!("wavelet function"))


samp = FiniteArray(-1, rand(4), Zero)
plot(-7:8, samp[-7:8])
plot!(-1:2, samp.values)
display(title!("padding: Zero"))

samp.padding = Cycle
plot(-7:8, samp[-7:8])
plot!(-1:2, samp.values)
display(title!("padding: Cycle"))

samp.padding = Linear
plot(-7:8, samp[-7:8])
plot!(-1:2, samp.values)
display(title!("padding: Linear"))

samp.padding = SymmetryInt
plot(-7:8, samp[-7:8])
plot!(-1:2, samp.values)
display(title!("padding: SymmetryInt"))

samp.padding = SymmetryHalf
plot(-7:8, samp[-7:8])
plot!(-1:2, samp.values)
display(title!("padding: SymmetryHalf"))
