module MetalExt

import ChainRulesCore
import ChainRulesCore: NoTangent
import Flux
import Flux: FluxCPUAdaptor, FluxMetalAdaptor, _mtl, _isleaf, adapt_storage, fmap
import Flux: DenseConvDims, Conv, ConvTranspose, Dense, conv, conv_reshape_bias
import NNlib

using Metal
using Adapt
using Random
using Zygote

const USE_Metal = Ref{Union{Nothing, Bool}}(nothing)

function check_use_metal()
    isnothing(USE_Metal[]) || return

    USE_Metal[] = Metal.functional()
    if !USE_Metal[]
        @info """
        The Metal function is being called but Metal is not functional.
        Defaulting back to the CPU. (No action is required if you want to run on the CPU).
        """ maxlog=1
    end
    return
end
ChainRulesCore.@non_differentiable check_use_metal()

include("functor.jl")
include("batchnorm.jl")
include("conv.jl")

function __init__()
    Flux.Metal_LOADED[] = true
end

# TODO
# fail early if input to the model is not on the device (e.g. on the host)
# otherwise we get very cryptic errors & segfaults at the rocBLAS level

end
