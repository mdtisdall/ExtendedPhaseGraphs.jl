# # Spin States
module SpinStates


## Individual spin states
# Spin states, when represeted as a single object, are simply a tuple with the
# the $F^+$, $F^-$, and $Z$ components available as complex values.
struct State{T<:Complex}
	fPlus::T
	fMinus::T
	z::T
end

## Aggregate spin states
# The total of all $N$ spin states, each representing $M$ spin conditions, are
# represented as an $M \times N \times 3$ array, organized in memory such that
# all $F^+$ entries go first, followed by all $F^-$ entries, and finally all
# $Z$ entries.
#
# Also, in the name of efficiency, we keep two buffers: a front buffer that
# represents the current state of the system, and a back buffer that can be used
# as the destination for efficient out-of-place operations mutating the state of
# the system. We can then just do a buffer-swap via exchanging pointers to these
# buffers, without requiring us to copy the whole system state. 
struct States{T<:Complex, AT<:AbstractArray{T, 3}}
	buffers::Vector{AT}
	States{T, AT}(m::Int, n::Int) where
		{T<:Complex, AT<:AbstractArray} =
			new([zeros(T, m, n, 3), zeros(T, m, n, 3)])
end

# We can do that actual buffer-swap by just reversing the order of the buffers
function swapbuffers!(s::States{T, AT}) where
		{T<:Complex, AT<:AbstractArray}
	reverse(s.buffers)
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
		reshapedBackBuffer = reshape(s.buffers[2], (:,3))
		reshapedBackBuffer = reshape(s.buffers[1], (:,3)) * f.opMat
		swapbuffers!(s)
end

end
