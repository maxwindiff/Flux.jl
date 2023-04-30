# Convert Float64 to Float32, but preserve Float16.
adapt_storage(::FluxMetalAdaptor, x::T) where T <: AbstractArray =
    isbits(x) ? x : MtlArray(x)
adapt_storage(::FluxMetalAdaptor, x::AbstractArray{T, N}) where {T <: AbstractFloat, N} =
    isbits(x) ? x : MtlArray{Float32, N}(x)
adapt_storage(::FluxMetalAdaptor, x::AbstractArray{Float16, N}) where N =
    isbits(x) ? x : MtlArray{Float16, N}(x)

adapt_storage(::FluxMetalAdaptor, x::Zygote.FillArrays.AbstractFill) =
    MtlArray(collect(x))
adapt_storage(::FluxMetalAdaptor, x::Zygote.OneElement) = MtlArray(collect(x))
# adapt_storage(::FluxMetalAdaptor, x::Random.TaskLocalRNG) =
#     AMDGPU.rocRAND.default_rng()
# adapt_storage(::FluxMetalAdaptor, x::AMDGPU.rocRAND.RNG) = x
# adapt_storage(::FluxMetalAdaptor, x::AbstractRNG) = error("""
#     Cannot map RNG of type $(typeof(x)) to AMDGPU.
#     AMDGPU execution only supports Random.default_rng().""")

# adapt_storage(::FluxCPUAdaptor, x::AMDGPU.rocRAND.RNG) = Random.default_rng()

function ChainRulesCore.rrule(
    ::typeof(Adapt.adapt_storage), to::FluxCPUAdaptor, x::Metal.MtlArray,
)
    adapt_storage(to, x), dx -> (
        NoTangent(), NoTangent(),
        adapt_storage(FluxMetalAdaptor(), unthunk(dx)))
end

function _mtl(x)
    check_use_metal()
    USE_Metal[] || return x
    fmap(x -> Adapt.adapt(FluxMetalAdaptor(), x), x; exclude=_isleaf)
end

# Since MIOpen supports only cross-correlation as convolution,
# for the actual convolution, we flip horizontally and vertically the weights.
# Same for CPU -> GPU & GPU -> CPU movements.
# Note, that gradients are also flipped.

# CPU -> GPU

_conv_basetype(c::Type{C}) where C <: Conv = Conv
_conv_basetype(c::Type{C}) where C <: ConvTranspose = ConvTranspose

function adapt_storage(to::FluxMetalAdaptor, m::C) where C <: Union{Conv, ConvTranspose}
    flipped_weight = reverse(m.weight; dims=ntuple(i -> i, ndims(m.weight) - 2))
    _conv_basetype(C)(
        Adapt.adapt(to, m.σ),
        Adapt.adapt(to, flipped_weight),
        Adapt.adapt(to, m.bias),
        m.stride, m.pad, m.dilation, m.groups)
end

# Don't adapt again.
function adapt_storage(
    to::FluxMetalAdaptor, m::Conv{N, M, F, A, V},
) where {N, M, F, A <: MtlArray, V}
    return m
end

function adapt_storage(
    to::FluxMetalAdaptor, m::ConvTranspose{N, M, F, A, V},
) where {N, M, F, A <: MtlArray, V}
    return m
end

_mtl(m::Union{Conv, ConvTranspose}) = adapt_storage(FluxMetalAdaptor(), m)

# GPU -> CPU

function Flux.cpu(m::Conv{N, M, F, A, V}) where {N, M, F, A <: MtlArray, V}
    adapt_storage(FluxCPUAdaptor(), m)
end

function Flux.cpu(m::ConvTranspose{N, M, F, A, V}) where {N, M, F, A <: MtlArray, V}
    adapt_storage(FluxCPUAdaptor(), m)
end

function adapt_storage(
    to::FluxCPUAdaptor, m::Conv{N, M, F, A, V},
) where {N, M, F, A <: MtlArray, V}
    dims = ntuple(i -> i, ndims(m.weight) - 2)
    Conv(
        Adapt.adapt(to, m.σ), reverse(Adapt.adapt(to, m.weight); dims),
        Adapt.adapt(to, m.bias), m.stride, m.pad, m.dilation, m.groups)
end

function adapt_storage(
    to::FluxCPUAdaptor, m::ConvTranspose{N, M, F, A, V},
) where {N, M, F, A <: MtlArray, V}
    dims = ntuple(i -> i, ndims(m.weight) - 2)
    ConvTranspose(
        Adapt.adapt(to, m.σ), reverse(Adapt.adapt(to, m.weight); dims),
        Adapt.adapt(to, m.bias), m.stride, m.pad, m.dilation, m.groups)
end


function adapt_storage(
    to::FluxMetalAdaptor, m::Dense{F, M, B},
) where {F, M <: MtlMatrix, B}
    println("in Metal.Dense:metal")
    return m
end
    
function adapt_storage(
    to::FluxCPUAdaptor, m::Dense{F, M, B},
) where {F, M <: MtlMatrix, B}
    println("in Metal.Dense:cpu")
    Dense(Adapt.adapt(to, m.F), Adapt.adapt(to, m.M), Adapt.adapt(to, m.B))
end
    