"""
    sqrt(str[; precision])

Calculate the square root of positive fixed-point number.

# Examples
```julia-repl
julia> sqrt("11001")
"101"

julia> sqrt("1010110.01101"; precision = 10)
"1001.0100101110"
```
"""
function sqrt(str::AbstractString; precision::Int = 0)
    bits, len_decimal = String2Bits(str)
    n_output_bits = (length(bits) - len_decimal) ÷ 2 + precision
    n_output_bits < 1 && return ""
    return Bits2String(binarysqrt(bits, n_output_bits), precision)
end


function binarysqrt(bits, n_bits::Int)
    bits = reverse(bits)
    quotient = BitVector()
    remainder = BitVector()
    for _ ∈ 1:n_bits
        if length(bits) > 0
            push!(remainder, pop!(bits))
            push!(remainder, pop!(bits))
        else
            push!(remainder, 0)
            push!(remainder, 0)
        end                                         # remainder = 4 * remainder + p
        tmp1 = vcat(.~quotient, BitVector([1, 1]))  # tmp1 = -(4 * quotient + 1)

        # padding `remainder` and `tmp1` to the same length
        max_len = max(length(remainder), length(tmp1))
        remainder = vcat(falses(max_len - length(remainder)), remainder)
        tmp1 = vcat(trues(max_len - length(tmp1)), tmp1)

        carry, sum_result = binaryadd(remainder, tmp1)  # carry = remainder ≥ -tmp1; sum_result = remainder + tmp1
        push!(quotient, carry)
        remainder = carry ? sum_result : remainder
    end
    return quotient
end

function binaryadd(a, b)
    result = BitVector()
    carry = false
    for idx ∈ length(a):-1:1
        # simulating the full-adder
        t = a[idx] ⊻ b[idx]
        push!(result, t ⊻ carry)
        carry = (a[idx] & b[idx]) | (t & carry)
    end
    return carry, reverse(result)
end



function IsBinaryNumberString(str)
    length(str) == 0 && return false
    get_dot = false
    for char ∈ str
        if char != '0' && char != '1'
            char != '.' && return false
            get_dot && return false
            get_dot = true
        end
    end
    return true
end


"""
    String2Bits(str[, checkstring])

Convert a binary string to a list of booleans, supports fixed-point numbers.

Return a list of booleans and a number indicating the length of decimal.
"""
function String2Bits(str; checkstring::Bool = true)
    checkstring && ~IsBinaryNumberString(str) && throw(ArgumentError("invalid binary number: " * str))
    integer, decimal = ('.' ∈ str) ? split(str, '.') : (str, "")
    isodd(length(integer)) && (integer = "0" * integer)
    isodd(length(decimal)) && (decimal = decimal * "0")
    return BitVector(map(char -> char == '1', collect(integer * decimal))), length(decimal)
end

"""
    Bits2String(bits[, len_decimal])

Convert a list of booleans to binary string.
"""
function Bits2String(bits, len_decimal::Integer = 0)
    str = string(map(bit -> bit ? '1' : '0', bits)...)
    len_decimal == 0 && return str
    len_decimal < 0 && return str * "0" ^ -len_decimal
    idx = length(str) - len_decimal
    return str[begin:idx] * "." * str[idx + 1:end]
end
