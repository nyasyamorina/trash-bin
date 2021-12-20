using Plots, Test, QuadGK



function HaarSimular(f::Function, j::Integer)   # in range [0,1)
    step = 1 / 2^j
    return f.((0:step:1-step) .+ 0.5step)
end

function HaarCall(V::AbstractVector, x::Number)
    idx = clamp(floor(Int, x * length(V)) + 1, 1, length(V))
    return V[idx]
end
function HaarCall(V::AbstractVector, x::AbstractVector)
    idx = clamp.(floor.(Int, x .* length(V)) .+ 1, 1, length(V))
    return V[idx]
end


function HaarDecompose(V::AbstractVector)
    tmp = reshape(V, (2, :))
    return (
        (tmp[1, :] .+ tmp[2, :]) ./ 2,
        (tmp[1, :] .- tmp[2, :]) ./ 2
    )
end

function HaarCompose(V::AbstractVector, W::AbstractVector)
    tmp1 = repeat(V, inner = (2,))
    tmp2 = repeat(reshape(W, (1, :)), outer = (2, 1))
    tmp2[2, :] .*= -1
    tmp3 = reshape(tmp2, (:,))
    return tmp1 .+ tmp3
end


function BreakDown(V::AbstractVector)
    j0::Int = log2(length(V))       # length(V) should be pow of 2
    result = similar(V)
    idx = 1
    V_j = V
    for j ∈ j0-1:-1:0
        shift = 2^j
        V_j, W = HaarDecompose(V_j)
        result[idx:idx + shift - 1] .= W
        idx += shift
    end
    result[idx:idx] .= V_j
    return result
end

function Refactor(WV::AbstractVector)
    idx = length(WV)
    j0::Int = log2(idx)             # length(WV) should be pow of 2
    V_j = WV[idx:idx]
    for j ∈ 1:j0
        shift = 2^(j - 1)
        idx -= shift
        V_j = HaarCompose(V_j, WV[idx:idx + shift - 1])
    end
    return V_j
end


function LowpassFilter(width::Real)
    j = floor(Int, -log2(width))
    return WV -> WV[length(WV) - 2^j + 1:end]
end

function Compress(rate::Real)
    return WV -> begin
        idx = clamp(floor(Int, length(WV) * rate), 1, length(WV))
        critical = sort(abs.(WV))[idx]
        filter(v -> v ≥ critical, WV)
        map(v -> (abs(v) ≥ critical) ? v : 0, WV)
    end
end


function HaarRelativeError(f::Function, V::AbstractVector)
    return √(
        quadgk(x -> (f(x) - HaarCall(V, x))^2, 0, 1)[1] /
        quadgk(x -> f(x)^2, 0, 1)[1]
    )
end



f(x) = begin
    t0 = 2π * x
    t1 = t0 - 1.2
    t2 = t1 - 4.5
    0.5sin(t0) + 0.49sin(2t0+2.5) + 0.51sin(4t0+0.7) + 0.16sin(8t0) -
        1.2exp(-100 * t1^2)*sin(16t1) + 1.2exp(-100 * t2^2)*sin(16t2)
end


V8 = HaarSimular(f, 8)

@test all((V8 |> BreakDown |> Refactor) .≈ V8)
# Because of floating point errors, `≈` should be used instead of `==`

Vfilted = V8 |> BreakDown |> LowpassFilter(1/16) |> Refactor
Vcompress80 = V8 |> BreakDown |> Compress(0.8) |> Refactor
Vcompress90 = V8 |> BreakDown |> Compress(0.9) |> Refactor

@show HaarRelativeError(f, V8)
@show HaarRelativeError(f, Vcompress80)
@show HaarRelativeError(f, Vcompress90)

x = 0:0.001:1
display(plot(x, f.(x); title = "origin signal"))
display(plot(x, HaarCall(V8, x); title = "simular with j = 8"))
display(plot(x, HaarCall(Vfilted, x); title = "filte wdith under 1/16 signal"))
display(plot(x, HaarCall(Vcompress80, x); title = "compress with rate = 80%"))
display(plot(x, HaarCall(Vcompress90, x); title = "compress with rate = 90%"))
