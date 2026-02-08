module NFFTReactantExt

using NFFT, NFFT.AbstractNFFTs
using NFFT: NFFTParams, 
            initParams, getWindow, shiftNodes!, checkNodes, TENSOR,
            precomputeWindowHatInvLUT, precompWindowHatInvLUT, precomputePolyInterp
using LinearAlgebra
using AbstractFFTs
using Reactant
using Reactant: TracedRArray, TracedRNumber, TraceMode, traced_type_inner
using ReactantCore: @trace

include("implementation.jl")

end