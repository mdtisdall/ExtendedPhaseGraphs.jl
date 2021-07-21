# # Spin States
module SpinStates

using LinearAlgebra

using StructArrays

## Individual spin states
# Spin states, when represeted as a single object, are simply a struct with the
# the $F^+$, $F^-$, and $Z$ components available as complex values.
struct State{T<:Complex}
    fPlus::T
    fMinus::T
    z::T
end

## Aggregate spin states
# The total of all $M$ spin states, each contained within $N$ spin conditions,
# are represented as an $M \times N \times 3$ array, organized in memory such
# that all $F^+$ entries go first, followed by all $F^-$ entries, and finally all
# $Z$ entries.
#
# Also, in the name of efficiency, we keep two buffers: a front buffer that
# represents the current state of the system, and a back buffer that can be used
# as the destination for efficient out-of-place operations mutating the state of
# the system. We can then just do a buffer-swap via exchanging pointers to these
# buffers, without requiring us to copy the whole system state. 
struct States{T<:Complex, AT<:AbstractArray{T, 3}}
    buffers::Vector{AT}
    originIndex::Int
    function States{T, AT}(m::Int, n::Int) where
        {T<:Complex, AT<:AbstractArray{T, 3}}
        @assert isodd(m) "m dimension of States must be odd"
        new([zeros(T, m, n, 3), zeros(T, m, n, 3)], 1 + ((m - 1) / 2))
    end
end

function fullyrelaxstates!(s::S) where
    {S <: States}
    s.buffers[1][s.originIndex,:,:] .= 1.0
    nothing
end



# We can do that actual buffer-swap by just reversing the order of the buffers
function swapbuffers!(s::States{T, AT}) where
        {T<:Complex, AT<:AbstractArray}
    reverse!(s.buffers)
    nothing
end
    
##Excitation operation
struct Excitation{T<:AbstractFloat}

    opMat::DenseArray{Complex{T}, 2}

    function Excitation(flipAngle::T, phase::T) where T<:AbstractFloat
        cosAlpha = cos(flipAngle)
        sinAlpha = sin(flipAngle)
        alphaDiv2 = flipAngle/2.0
        cosAlphaDiv2 = cos(alphaDiv2)
        cosAlphaDiv2Sq = cosAlphaDiv2^2
        sinAlphaDiv2 = sin(alphaDiv2)
        sinAlphaDiv2Sq = sinAlphaDiv2^2
        expPhase = exp(phase * (1.0im))
        expPhaseSq = exp(phase * (2.0im))
        expPhaseInv = exp(phase * (-1.0im))
        expPhaseInvSq = exp(phase * (-2.0im))

        tempMat = zeros(Complex{T}, 3, 3)
        tempMat[1] = cosAlphaDiv2Sq
        tempMat[2] = expPhaseSq * sinAlphaDiv2Sq
        tempMat[3] = (-1.0im) * expPhase * sinAlpha
        tempMat[4] = expPhaseInvSq * sinAlphaDiv2Sq
        tempMat[5] = cosAlphaDiv2Sq
        tempMat[6] = (1.0im) * expPhaseInv * sinAlpha
        tempMat[7] = (-0.5im) * expPhaseInv * sinAlpha
        tempMat[8] = (0.5im) * expPhase * sinAlpha
        tempMat[9] = cosAlpha
        new{T}(tempMat)
    end
end

function (f::Excitation)(s::States)
    #reshapedBackBuffer = reshape(s.buffers[2], (:,3))
    #reshapedBackBuffer = reshape(s.buffers[1], (:,3)) * f.opMat
    mul!(reshape(s.buffers[2],(:,3)), reshape(s.buffers[1], (:,3)), f.opMat)
    swapbuffers!(s)
    nothing
end

## Individual spin environments
# Spin environments, represented as a single object are just a struct
# with the relaxation parameters available as floating point values.
struct SpinEnvironment{T<:AbstractFloat}
    t1::T
    t2::T
end

struct Environments{T<:AbstractFloat}
    relaxationConstants::StructVector{SpinEnvironment{T}}
    function Environments{T}(e::Vector{SpinEnvironment{T}}) where
        T<:AbstractFloat
        new(StructVector{SpinEnvironment{T}}(
            reinterpret(reshape,T, e), dims=1)) 
    end  
end

struct SpinEnvironmentScales{T<:AbstractFloat}
    scaleT1::T
    scaleT2::T
    addsT1::T
end

struct Relaxation{T<:AbstractFloat}
    relaxationScales::StructVector{SpinEnvironmentScales{T}}
    function Relaxation{T}(e::Environments{T}, duration::T) where
        T<:AbstractFloat
        tempScaleT1 = exp.(-duration ./ e.relaxationConstants.t1)
        tempScaleT2 = exp.(-duration ./ e.relaxationConstants.t2)
        tempAddsT1 = 1.0 .- tempScaleT1 
        new(StructVector{SpinEnvironmentScales{T}}( 
            scaleT1=tempScaleT1,
            scaleT2=tempScaleT2,
            addsT1=tempAddsT1))
    end  
end
#
#function (f::Relaxation)(s::States)
#    s.buffers[2] = broadcast(*, f.relaxationScales, s.buffers[1])
#    view(s.buffers[2],(:,s.originIndex,3)) =
#        broadcast(+, f.relaxationAdds, view(s.buffers[1],(:,s.originIndex,3))) 
#end

end
