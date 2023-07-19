using Plots, QuadGK, SpecialFunctions
Plots.theme(:dark)
cd(@__DIR__)

# data from "https://www.lmfdb.org/zeros/zeta/"
include(raw"reimann-zeta-zeros\example-julia\ReimannZetaZeros.jl")




module PrimeFunctions
    export getprime, getprimecount, getmobius, getvonmangoldt, getchebyshev


    const is_prime = BitVector((false, true))
    const p = Vector{Int}()
    const π = Vector{Int}()
    const μ = Vector{Int8}()
    const Λ = Vector{Float64}()
    const ψ = Vector{Float64}()


    isprime(x::Integer) = begin; genisprime(x); is_prime[x]; end
    for (sym, list) ∈ ((:prime, :p), (:primecount, :π), (:mobius, :μ), (:vonmangoldt, :Λ), (:chebyshev, :ψ))
        getfun = Symbol("get", sym)
        genfun = Symbol("gen", sym)
        @eval $sym(x) = $list[floor(Int, x)]
        @eval $getfun(x::Integer) = begin; $genfun(x); $list[max(1, x)]; end
        @eval $getfun(x) = $getfun(floor(Int, x))
    end


    function genisprime(x::Integer)
        length(is_prime) ≥ x && return
        updateprimes()
        start = length(is_prime) + 1
        extra = trues(x - length(is_prime))
        max_p = floor(Int, sqrt(x))
        need_continue = true
        for prime ∈ p
            a = ceil(Int, start / prime)
            k = a * prime - start + 1
            extra[k:prime:end] .= false
            prime ≥ max_p && (need_continue = false; break)
        end
        if need_continue
            for k ∈ axes(extra, 1)
                if extra[k]
                    prime = k + start - 1
                    extra[(k + prime):prime:end] .= false
                end
                k ≥ max_p && break
            end
        end
        sizehint!(is_prime, x)
        append!(is_prime, extra)
    end

    function genprime(x::Integer)
        length(p) ≥ x && return
        bounds = length(is_prime)
        n = endprime() + 1
        sizehint!(p, x)
        while length(p) < x
            if n > bounds
                # alway has prime in range (x - 4/π √x log(x), x)
                # but f(x) = x - 4/π √x log(x) has no explit inverse function,
                # and x + 5/π √x log(x) > f^{-1}(x), x ≥ 2
                bounds += ceil(Int, 5/π * sqrt(bounds) * log(bounds))
                genisprime(bounds)
            end
            updateprimes(n)
            n = bounds + 1
        end
    end

    function genprimecount(x::Integer)
        length(π) ≥ x && return
        length(is_prime) < x && genisprime(x)
        (n, pi) = isempty(π) ? (0, 0) : (length(π), π[end])
        extra = Vector{Int}(undef, x - n)
        for k ∈ (n+1):x
            is_prime[k] && (pi += 1)
            extra[k - n] = pi
        end
        sizehint!(π, x)
        append!(π, extra)
    end

    function genmobius(x::Integer)
        length(μ) ≥ x && return
        start = length(μ) + 1
        extra = ones(Int8, x - length(μ))
        length(is_prime) < x && genisprime(x)   
        updateprimes()
        for prime ∈ p
            a = ceil(Int, start / prime)
            k = a * prime - start + 1
            extra[k:prime:end] .*= -1
            a = ceil(Int, start / prime^2)
            k = a * prime^2 - start + 1
            extra[k:(prime^2):end] .= 0
            prime ≥ x && break
        end
        sizehint!(μ, x)
        append!(μ, extra)
    end

    function genvonmangoldt(x::Integer)
        length(Λ) ≥ x && return
        start = length(Λ) + 1
        extra = zeros(x - length(Λ))
        length(is_prime) < x && genisprime(x)
        updateprimes()
        for prime ∈ p
            prime > x && break
            logp = log(prime)
            a = ceil(Int, log(start) / logp)
            a = max(a, 1)
            n = prime^a
            while n ≤ x
                extra[n - start + 1] = logp
                n *= prime
            end
        end
        sizehint!(Λ, x)
        append!(Λ, extra)
    end

    function genchebyshev(x::Integer)
        length(ψ) ≥ x && return
        length(Λ) < x && genvonmangoldt(x)
        (n, psi) = isempty(ψ) ? (0, 0.0) : (length(ψ), ψ[end])
        extra = Vector{Float64}(undef, x - n)
        for k ∈ (n+1):x
            psi += Λ[k]
            extra[k - n] = psi
        end
        sizehint!(ψ, x)
        append!(ψ, extra)
    end


    function updateprimes(low)
        for k ∈ low:length(is_prime)
            is_prime[k] && push!(p, k)
        end
    end
    updateprimes() = updateprimes(endprime() + 1)

    endprime() = isempty(p) ? 0 : p[end]
end

function Π(x::Real)
    PrimeFunctions.genprimecount(floor(Int, x))
    result = 0.
    max_n = floor(Int, 1 / log(x, 2))
    for n ∈ 1:max_n
        result += PrimeFunctions.primecount(x^(1 / n)) * (1 / n)
    end
    return result
end

function callsmooth(f, x::AbstractFloat)
    f₋ = f(prevfloat(x))
    f₊ = f(nextfloat(x))
    return (f₋ + f₊) / 2
end
callsmooth(f, x) = callsmooth(f, float(x))
# PrimeFunctions.getprimecount(7) = 4
# callsmooth(PrimeFunctions.getprimecount, 7) = 3.5



function SpecialFunctions.expinti(x::Complex)
    imag(x) > 0 && return -expint(-x) + im * π
    imag(x) < 0 && return -expint(-x) - im * π
    real(x) > 0 && return -expint(-x) - im * π
                   return -expint(-x)
end

li(x, ρ) = expinti(log(x) * ρ)  # li(x) = Ei(ln(x))
li(x) = li(x, 1)

sumtrivial_li(x) = quadgk(t -> inv(t * (t^2 - 1) * log(t)), x, Inf)[1]


function R(x, ρ; max_iter)
    result = zero(x)
    for n ∈ 1:max_iter
        μ = PrimeFunctions.mobius(n)
        μ == 0 && continue
        result += μ / n * expinti(log(x) * ρ / n)
    end
    return result
end
R(x; max_iter) = R(x, 1; max_iter)


function ψ₀(x; zeros)
    result = x - log(2π)
    result -= 0.5log(1 - 1 / x^2)
    for t ∈ zeros
        ρ = complex(0.5, t)
        result -= 2real(x^ρ / ρ)
    end
    return result
end

function Π₀(x; zeros)
    result = li(x) - log(2)
    result += sumtrivial_li(x)
    for t ∈ zeros
        ρ = complex(0.5, t)
        result -= 2real(li(x, ρ))
    end
    return result
end

function π₀(x; zeros, max_iter = 7)
    PrimeFunctions.genmobius(max_iter)
    result = R(x; max_iter)
    for n ∈ 1:max_iter
        μ = PrimeFunctions.mobius(n)
        μ == 0 && continue
        result += μ / n * sumtrivial_li(x^(1/n))
    end
    for t ∈ zeros
        result -= 2real(R(x, complex(0.5, t); max_iter))
    end
    return result
end





log10_xs = 1:0.01:9
xs = 10 .^ log10_xs

PrimeFunctions.genprimecount(floor(Int, maximum(xs)))
f1 = PrimeFunctions.primecount.(xs)
f2 = xs ./ log.(xs)
f3 = li.(xs)
e1 = abs.(f1 .- f2) ./ f1
e2 = abs.(f1 .- f3) ./ f1
plot(xs, [e1 e2]; label = ["rel err of `x/ln x`" "rel err of `li(x)`"], xscale = :log10, minorgrid = true)
display(ylims!((0, 0.5)))




xs = 1.8:0.1:100
zeros = Float64.(ReimannZetaZeros.getzeros(25000))

ψ₀xs = callsmooth.(PrimeFunctions.getchebyshev, xs)
rough_ψ₀xs = ψ₀.(xs; zeros = zeros[1:100])
fine_ψ₀xs  = ψ₀.(xs; zeros)
display(plot(xs, [ψ₀xs rough_ψ₀xs]; label = ["ψ₀" "explicit ψ₀"]))
display(plot(xs, abs.(ψ₀xs .- fine_ψ₀xs); label = "abs err of explicit ψ₀", legend = :topleft))

Π₀xs = callsmooth.(Π, xs)
rough_Π₀xs = Π₀.(xs; zeros = zeros[1:100])
fine_Π₀xs  = Π₀.(xs; zeros)
display(plot(xs, [Π₀xs rough_Π₀xs]; label = ["Π₀" "explicit Π₀"]))
display(plot(xs, abs.(Π₀xs .- fine_Π₀xs); label = "abs err of explicit Π₀", legend = :topleft))

π₀xs = callsmooth.(PrimeFunctions.getprimecount, xs)
rough_π₀xs = π₀.(xs; zeros = zeros[1:100])
fine_π₀xs  = π₀.(xs; zeros)
display(plot(xs, [π₀xs rough_π₀xs]; label = ["π₀" "explicit π₀"]))
display(plot(xs, abs.(π₀xs .- fine_π₀xs); label = "abs err of explicit π₀", legend = :topleft))