using ColorTypes, FixedPointNumbers


Base.zero(::Type{NTuple{4, UInt8}}) = (0x0, 0x0, 0x0, 0x0)

function _toT4(c::Colorant)
    argb = ARGB32(c).color
    return UInt8.((argb >> 16, argb >> 8, argb, argb >> 24) .& 0xFF)
end

RGB{N0f8}(c::NTuple{4, UInt8}) = RGB(reinterpret.(N0f8, c[1:3])...)
RGBA{N0f8}(c::NTuple{4, UInt8}) = RGBA(reinterpret.(N0f8, c)...)

# index in julia starts at 1, instead 0
_index_position((r, g, b, a)) = (3r + 5g + 7b + 11a) % 64 + 1


@inbounds function QOIsave(path, im::AbstractMatrix{T}, colorspace::Symbol = :linear) where T <: Colorant
    @assert all((size(im) .>> 32) .== 0) "the `qoi` format only accept image which both width and height is less than 2^32"
    @assert colorspace ∈ (:sRGB, :linear) "the `qoi` format only accept `:sRGB` or `:linear` color space"

    channels = T <: TransparentColor ? 0x4 : 0x3
    open(path; write = true) do file
        # qoi_header: {"qoif", width, height, channels, colorspace} 14-byte
        write(file, "qoif", hton(UInt32(size(im, 2))), hton(UInt32(size(im, 1))))
        write(file, channels, colorspace ≠ :linear ? 0x0 : 0x1)

        curr = (0x0, 0x0, 0x0, 0xFF)            # current pixel, use BLACK as starting pixel
        running = zeros(typeof(curr), 64)       # the runinng array for QOI_OP_INDEX encoding
        running_idx = 1                         # the runinng index of current pixel
        pixels = reshape(transpose(im), :)      # transform image to pixel vector
        total_pixels = length(pixels)
        pixel_idx = 0                           # the current pixel index

        while pixel_idx < total_pixels
            prev = curr
            running[running_idx] = prev         # set previous pixel into runinng array

            curr_color = pixels[pixel_idx += 1] # get current pixel
            curr = _toT4(curr_color)            # transform pixel color to tuple for calculating
            running_idx = _index_position(curr)

            if curr == prev                                         # -> QOI_OP_RUN
                run = 0x0
                pixel_idx += 1
                while pixel_idx < total_pixels && pixels[pixel_idx] == curr_color && run < 0x3D
                    run += 0x1
                    pixel_idx += 1
                end
                pixel_idx -= 1
                write(file, 0xC0 | run)                 # 11[6-bit run-1]
                continue

            elseif curr == running[running_idx]                     # -> QOI_OP_INDEX
                write(file, UInt8(running_idx - 1))     # 00[6-bit index]
                continue

            elseif curr[4] == prev[4]
                d = curr[1:3] .- prev[1:3] .+ 0x2
                if all(d .< 0x4)                                     # -> QOI_OP_DIFF
                    write(file, 0x40 | (d[1] << 4) | (d[2] << 2) | d[3])    # 01[2-bit dr][2-bit dg][2-bit db]
                    continue

                elseif (d = (d[1] - d[2], d[2] + 0x16, d[3] - d[2]) .+ 0x8;
                        d[2] < 0x40 && d[1] < 0x10 && d[3] < 0x10)  # -> QOI_OP_LUMA
                    write(file, 0x80 | d[2], (d[1] << 4) | d[3])    # 10[6-bit dg] | [4-bit dr-dg][4-bit db-dg]
                    continue
                end
            end

            # QOI_OP_RGB:  11111110 | [8-bit r] | [8-bit g] | [8-bit b]
            # QOI_OP_RGBA: 11111111 | [8-bit r] | [8-bit g] | [8-bit b] | [8-bit a]
            write(file, 0xFB + channels, curr[1:channels]...)
        end

        write(file, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x1);    # qoi_end
    end
end


function QOIload(path)
    open(path; read = true) do file
        # read the header
        magic = String(read(file, 4))
        width = ntoh(read(file, UInt32))
        height = ntoh(read(file, UInt32))
        channels = read(file, UInt8)
        colorspace = read(file, UInt8)

        #@assert magic == "qoif"         "actually the magic number of the file is not important"
        @assert channels ∈ (0x3, 0x4)    "only support channels 3: RGB, 4: RGBA, got $(channels)"
        #@assert colorspace ∈ (0x0, 0x1) "actually the colorspace will not used"

        # split decoding into functions to ensure type stability for speed
        _qoi_load(channels == 0x3 ? RGB{N0f8} : RGBA{N0f8}, file, height, width)

        # the following line should not be uncommented
        #@assert all(read(file, 7) .== 0x0) && read(file, UInt8) == 0x1 "the end of file is not important"
    end
end

@inbounds function _qoi_load(::Type{T}, file, height, width) where T <: Union{RGB{N0f8}, RGBA{N0f8}}
    channels = T <: RGB ? 0x3 : 0x4
    total_pixels = height * width
    pixels = Vector{T}(undef, total_pixels)
    pixel_idx = 0

    curr = (0x0, 0x0, 0x0, 0xFF)
    running = zeros(typeof(curr), 64)

    while pixel_idx < total_pixels
        byte = read(file, UInt8)
        tag = byte & 0xC0
        data = byte & 0x3F

        if tag == 0xC0
            if byte - channels == 0xFB          # -> QOI_OP_RGB(A)
                c = read(file, channels)
                curr = (c[1], c[2], c[3], channels == 0x3 ? 0xFF : c[4])

            else                                # -> QOI_OP_RUN
                #@assert data < 0x3E "invalid run, it should be < 0x3E"
                pixel_idx += 1
                curr_color = T(curr)
                while pixel_idx < total_pixels && data > 0x0
                    pixels[pixel_idx] = curr_color
                    pixel_idx += 1
                    data -= 0x1
                end
                pixel_idx -= 1
            end

        elseif tag == 0x0                       # -> QOI_OP_INDEX
            curr = running[data + 1]

        else
            d = if tag == 0x40                  # -> QOI_OP_DIFF
                (data >> 4, (data >> 2) & 0x3, data & 0x3) .- 0x2

            else                                # -> QOI_OP_LUMA
                byte = read(file, UInt8)
                (byte >> 4, 0x8, byte & 0xF) .+ data .- 0x28
            end

            curr = ((curr[1:3] .+ d)..., curr[4])
        end

        pixels[pixel_idx += 1] = T(curr)
        running[_index_position(curr)] = curr
    end

    return collect(transpose(reshape(pixels, width, height)))
end
